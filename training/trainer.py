import torch
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm

from misc.utils import AverageMeter
from models.model_factory import model_factory, copy_frozen_model
from datasets.dataset_utils import make_dataloader
from losses.loss_factory import make_pr_loss
from eval.eval_utils import get_latent_vectors, euclidean_distance, cosine_dist
from datasets.forget_event import update_forget_events

from torchpack.utils.config import configs


class Trainer:
    def __init__(self, logger, train_environment):
        # Initialise inputs
        self.debug = configs.debug
        self.logger = logger
        self.epochs = configs.train.optimizer.epochs

        # Set up meters and stat trackers
        self.loss_pr_meter = AverageMeter()
        self.loss_contrasive_meter = AverageMeter()
        self.num_triplets_meter = AverageMeter()
        self.non_zero_triplets_meter = AverageMeter()
        self.embedding_norm_meter = AverageMeter()

        # Make dataloader
        self.dataloader = make_dataloader(pickle_file = train_environment, scene_id=0, memory = None)
        self.forget_events = {}

        # Build model
        assert torch.cuda.is_available, 'CUDA not available.  Make sure CUDA is enabled and available for PyTorch'
        self.model = model_factory(ckpt = None, use_prompt=False, use_scene_id=False, device = 'cuda')

        # Make loss functions
        self.loss_fn_pr = make_pr_loss()
   
    def make_optimizer(self):
        if configs.train.optimizer.weight_decay is None or configs.train.optimizer.weight_decay == 0:
            self.optimizer = torch.optim.Adam(self.meters(), lr=configs.train.optimizer.lr)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=configs.train.optimizer.lr, weight_decay=configs.train.optimizer.weight_decay)

    def make_scheduler(self):
        if configs.train.optimizer.scheduler is None:
            self.scheduler = None
        else:
            if configs.train.optimizer.scheduler == 'CosineAnnealingLR':
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=configs.train.optimizer.epochs+1,
                                                                    eta_min=configs.train.optimizer.min_lr)
            elif configs.train.optimizer.scheduler == 'MultiStepLR':
                if not isinstance(configs.train.optimizer.scheduler_milestones, list):
                    configs.train.optimizer.scheduler_milestones = [configs.train.optimizer.scheduler_milestones]
                self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, configs.train.optimizer.scheduler_milestones, gamma=0.1)
            else:
                raise NotImplementedError('Unsupported LR scheduler: {}'.format(configs.train.optimizer.scheduler))
    
    def before_epoch(self, epoch):
        # Reset meters
        self.loss_pr_meter.reset()
        self.loss_contrasive_meter.reset()
        self.num_triplets_meter.reset()
        self.non_zero_triplets_meter.reset()
        self.embedding_norm_meter.reset()

    def training_step(self, batch1, batch2, positives_mask, negatives_mask):

        n_positives = torch.sum(positives_mask).item()
        n_negatives = torch.sum(negatives_mask).item()
        if n_positives == 0 or n_negatives == 0:
            # Skip a batch without positives or negatives
            print('WARNING: Skipping batch without positive or negative examples')
            return None 

        # Get embeddings and Loss (embeddings stop gradients)
        self.optimizer.zero_grad()
        batch1 = {x: batch1[x].to('cuda') if x!= 'coords' else batch1[x] for x in batch1}
        embeddings1, projections1 = self.model(batch1)  # B x 256, B x *

        # place recognition loss
        loss_place_rec, num_triplets, non_zero_triplets, embedding_norm = self.loss_fn_pr(embeddings1, positives_mask, negatives_mask)
        loss_total = loss_place_rec
        
        # contrasitive loss
        loss_mi = torch.zeros(1)
        
        # Backwards
        loss_total.backward()
        self.optimizer.step()
        torch.cuda.empty_cache() # Prevent excessive GPU memory consumption by SparseTensors

        # Stat tracking
        self.loss_pr_meter.update(loss_place_rec.item())
        self.loss_contrasive_meter.update(loss_mi.item())
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
            ratio_non_zeros = self.non_zero_triplets_meter.avg / self.num_triplets_meter.avg
            if ratio_non_zeros < configs.train.batch_expansion_th:
                self.dataloader.batch_sampler.expand_batch()

        # Tensorboard plotting
        self.logger.add_scalar(f'Place_Rec_Loss', self.loss_pr_meter.avg, epoch)
        self.logger.add_scalar(f'Non_Zero_Triplets', self.non_zero_triplets_meter.avg, epoch)
        self.logger.add_scalar(f'Embedding_Norm', self.embedding_norm_meter.avg, epoch)
        
        # update forget events
        # if epoch % 2 == 0:
        model_latest = copy_frozen_model(self.model)
        update_forget_events(model_latest, self.dataloader.dataset.queries, self.forget_events)

    def train(self,):
        # Make optimizer
        self.make_optimizer()
        # Scheduler
        self.make_scheduler()
        self.model.prepare_before_train(stage=2)  # do not train the prompt module
        for epoch in tqdm(range(1, self.epochs + 1)):
            self.before_epoch(epoch)
            for idx, (batch1, batch2, positives_mask, negatives_mask) in enumerate(self.dataloader):
                self.training_step(batch1, batch2, positives_mask, negatives_mask)
                if self.debug and idx > 2:
                    break
            self.after_epoch(epoch)
            if self.debug and epoch > 2:
                break
        
        return self.model
