import os

import torch
import torch.nn.functional as F 

import numpy as np
import copy
from tqdm import tqdm 

from misc.utils import AverageMeter
from models.model_factory import model_factory, copy_frozen_model
from datasets.dataset_utils import make_dataloader
from losses.loss_factory import make_pr_loss, make_inc_loss
from eval.eval_utils import get_latent_vectors, euclidean_distance, cosine_dist
from datasets.forget_event import update_forget_events

from torchpack.utils.config import configs


class TrainerIncremental:
    def __init__(self, logger, memory, new_environment_pickle, pretrained_checkpoint, env_idx):
        # Initialise inputs
        self.debug = configs.debug 
        self.logger = logger 
        self.env_idx = env_idx

        # Set up meters and stat trackers 
        self.loss_total_meter = AverageMeter()
        self.loss_pr_meter = AverageMeter()
        self.loss_inc_meter = AverageMeter()
        self.num_triplets_meter = AverageMeter()
        self.non_zero_triplets_meter = AverageMeter()
        self.embedding_norm_meter = AverageMeter()

        # Make dataloader
        self.dataloader = make_dataloader(pickle_file = new_environment_pickle, scene_id=env_idx, memory = memory)
        self.forget_events = {}

        # Build models and init from pretrained_checkpoint
        assert torch.cuda.is_available, 'CUDA not available.  Make sure CUDA is enabled and available for PyTorch'
        self.model_old = model_factory(ckpt = pretrained_checkpoint, device = 'cuda')
        self.model_new = model_factory(ckpt = pretrained_checkpoint, device = 'cuda')
        self.model_new.use_prompt = configs.model.use_prompt
        self.model_new.promptor.qformer.use_scene_id = configs.model.use_scene_id

        # Make optimizer
        self.make_optimizer()
        
        # Scheduler
        self.make_scheduler()

        # Make loss functions
        self.loss_fn_pr = make_pr_loss()
        self.loss_fn_inc = make_inc_loss()
        if configs.train.loss.incremental.name == 'EWC':
            self.fisher_matrix, self.old_parameters = self.loss_fn_inc.get_fisher_matrix(
                                                        dataloader = self.dataloader,
                                                        model = self.model_new, 
                                                        optimizer = self.optimizer,
                                                        loss_fn = self.loss_fn_pr
                                                        )
    
    def get_optimizer_cfg(self, stage=-1):
        if stage == 1:  # training QFormer only
            cfg = configs.train.optimizer1
        elif stage == 2:  # fine tune modules except QFormer, learning rate should be smaller!
            cfg = configs.train.optimizer2
        else:
            cfg = configs.train.optimizer1
        return cfg
    
    def make_optimizer(self, stage=-1):
        cfg = self.get_optimizer_cfg(stage)
        if cfg.weight_decay is None or cfg.weight_decay == 0:
            self.optimizer = torch.optim.Adam(self.meters(), lr=cfg.lr)
        else:
            self.optimizer = torch.optim.Adam(self.model_new.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        return self.optimizer

    def make_scheduler(self, stage=-1):
        cfg = self.get_optimizer_cfg(stage)
        if cfg.scheduler is None:
            self.scheduler = None
        else:
            if cfg.scheduler == 'CosineAnnealingLR':
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=cfg.epochs+1,
                                                                    eta_min=cfg.min_lr)
            elif cfg.scheduler == 'MultiStepLR':
                if not isinstance(cfg.scheduler_milestones, list):
                    cfg.scheduler_milestones = [cfg.scheduler_milestones]
                self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, cfg.scheduler_milestones, gamma=0.1)
            else:
                raise NotImplementedError('Unsupported LR scheduler: {}'.format(cfg.scheduler))
        return self.scheduler
    
    def before_epoch(self, epoch):
        # Reset meters
        self.loss_total_meter.reset()
        self.loss_pr_meter.reset()
        self.loss_inc_meter.reset()
        self.num_triplets_meter.reset()
        self.non_zero_triplets_meter.reset()
        self.embedding_norm_meter.reset()
        
        # Adjust weight of incremental loss function if constraint relaxation enabled
        self.loss_fn_inc.adjust_weight(epoch)

    def training_step(self, batch1, batch2, positives_mask, negatives_mask):
        
        n_positives = torch.sum(positives_mask).item()
        n_negatives = torch.sum(negatives_mask).item()
        if n_positives == 0 or n_negatives == 0:
            # Skip a batch without positives or negatives
            print('WARNING: Skipping batch without positive or negative examples')
            return None
        # batch size: new samples (from current dataset), memory samples
        B_current = batch1['cloud'].shape[0] // 2

        # Get embeddings and Loss
        self.optimizer.zero_grad()
        batch1 = {x: batch1[x].to('cuda') if x!= 'coords' else batch1[x] for x in batch1}
        with torch.no_grad():
            embeddings1_old, _ = self.model_old(batch1)
        embeddings1_new, _ = self.model_new(batch1)
        
        # place recognition loss
        loss_place_rec, num_triplets, non_zero_triplets, embedding_norm = self.loss_fn_pr(embeddings1_new, positives_mask, negatives_mask)
        loss_total = loss_place_rec
        
        # knowledge distillation loss: compare features of the same samples embedded by new and old models, samples are from memory
        loss_incremental = torch.zeros(1)
        if configs.train.memory.num_pairs > 0:
            if configs.train.loss.incremental.name != 'EWC':  # StructureAware, DistributionAware
                loss_incremental = self.loss_fn_inc(embeddings1_old[B_current:], embeddings1_new[B_current:])
            else:
                loss_incremental = self.loss_fn_inc(self.model_new, self.old_parameters, self.fisher_matrix)
            loss_total += loss_incremental

        # Backwards
        loss_total.backward()
        self.optimizer.step()
        torch.cuda.empty_cache() # Prevent excessive GPU memory consumption by SparseTensors

        # Stat tracking
        self.loss_total_meter.update(loss_total.item())
        self.loss_pr_meter.update(loss_place_rec.item())
        self.loss_inc_meter.update(loss_incremental.item())
        self.num_triplets_meter.update(num_triplets)
        self.non_zero_triplets_meter.update(non_zero_triplets)
        self.embedding_norm_meter.update(embedding_norm)

        return None

    def get_forget_events(self):
        return self.forget_events

    def after_epoch(self, epoch):
        # Scheduler 
        if self.scheduler is not None:
            self.scheduler.step()

        # Dynamic Batch Expansion
        if configs.train.batch_expansion_th is not None:
            if self.num_triplets_meter.avg > 0:
                ratio_non_zeros = self.non_zero_triplets_meter.avg / self.num_triplets_meter.avg 
            else:
                ratio_non_zeros = 0
            if ratio_non_zeros < configs.train.batch_expansion_th:
                self.dataloader.batch_sampler.expand_batch()

        # Tensorboard plotting 
        self.logger.add_scalar(f'Step_{self.env_idx}/Total_Loss', self.loss_total_meter.avg, epoch)
        self.logger.add_scalar(f'Step_{self.env_idx}/Place_Rec_Loss', self.loss_pr_meter.avg, epoch)
        self.logger.add_scalar(f'Step_{self.env_idx}/Incremental_Loss', self.loss_inc_meter.avg, epoch)
        self.logger.add_scalar(f'Step_{self.env_idx}/Non_Zero_Triplets', self.non_zero_triplets_meter.avg, epoch)
        self.logger.add_scalar(f'Step_{self.env_idx}/Embedding_Norm', self.embedding_norm_meter.avg, epoch)
        
        # update forget events
        if configs.train.memory.use_greedy:  # epoch % 2 == 0 and 
            model_latest = copy_frozen_model(self.model_new)
            update_forget_events(model_latest, self.dataloader.dataset.queries, self.forget_events)

    def train(self, stage):
        print(f'Training Stage: {stage}')
        self.forget_events = {}
        cfg = self.get_optimizer_cfg(stage)
        self.model_new.train()
        # Make optimizer
        self.make_optimizer(stage)
        # Scheduler
        self.make_scheduler(stage)
        # stage 1: train QFormer;  stage 2: train except QFormer
        if stage == -1:
            stage == 2
        self.model_new.prepare_before_train(stage)
        for epoch in tqdm(range(1, cfg.epochs + 1)):
            self.before_epoch(epoch)
            for idx, (batch1, batch2, positives_mask, negatives_mask) in enumerate(self.dataloader):
                self.training_step(batch1, batch2, positives_mask, negatives_mask)
                if self.debug and idx > 2:
                    break
            self.after_epoch(epoch)
            if self.debug and epoch > 2:
                break
        
        return self.model_new
    
    def train_supersede(self):
        ''' Train supersedelt when using prompt, sth wrong with the code, need to fix! '''
        cfg = self.get_optimizer_cfg(stage=1)
        self.model_new.train()
        # optimizer and scheduler 1 / 2
        optimizer1, scheduler1 = self.make_optimizer(stage=1), self.make_scheduler(stage=1)
        optimizer2, scheduler2 = self.make_optimizer(stage=2), self.make_scheduler(stage=2)
        # training
        for epoch in tqdm(range(1, cfg.epochs + 1)):
            
            # stage 1
            self.before_epoch(epoch)
            self.optimizer = optimizer1
            self.scheduler = scheduler1
            self.model_new.prepare_before_train(stage=1)
            for idx, (batch1, batch2, positives_mask, negatives_mask) in enumerate(self.dataloader):
                self.training_step(batch1, batch2, positives_mask, negatives_mask)
                if self.debug and idx > 2:
                    break
            self.after_epoch(epoch)
            
            # stage 2
            self.before_epoch(epoch)
            self.optimizer = optimizer2
            self.scheduler = scheduler2
            self.model_new.prepare_before_train(stage=2)
            for idx, (batch1, batch2, positives_mask, negatives_mask) in enumerate(self.dataloader):
                self.training_step(batch1, batch2, positives_mask, negatives_mask)
                if self.debug and idx > 2:
                    break
            self.after_epoch(epoch)
            
            if self.debug and epoch > 2:
                break
        
        return self.model_new
