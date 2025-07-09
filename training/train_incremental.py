import os 
import torch 
import argparse 
import random
import datetime
import numpy as np

from torchpack.utils.config import configs 
from misc.utils import load_pickle, save_pickle
from datasets.memory import Memory
from models.model_factory import model_factory

from eval.evaluate import evaluate 
from eval.metrics import IncrementalTracker 
from trainer_incremental import TrainerIncremental

from torch.utils.tensorboard import SummaryWriter


if __name__ == '__main__':

    # Repeatability 
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    # Get args and configs 
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type = str, required = True, help = 'Path to configuration YAML file')

    args, opts = parser.parse_known_args()
    configs.load(args.config, recursive = True)
    configs.update(opts)
    
    inv = 'inv' in args.config
    
    if isinstance(configs.train.optimizer.scheduler_milestones, str):
        configs.train.optimizer.scheduler_milestones = [int(x) for x in configs.train.optimizer.scheduler_milestones.split(',')] # Allow for passing multiple drop epochs to scheduler
    print(configs)

    # Make save directory and logger
    model_dir = os.path.join(configs.save_dir, configs.model.name, 'models')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    now = datetime.datetime.now()
    name = now.strftime("%Y-%m-%d-%H-%M-%S")
    if configs.train.memory.use_greedy:
        name += '_select-greedy'
        if configs.train.memory.use_forget:
            name += '-forget'
        if configs.train.memory.use_dist:
            name += '-dist'
    else:
        name += '_select-random'
    name += f'{configs.train.memory.num_pairs}'
    if configs.train.memory.random_forget:
        name += '_forget-random'
    else:
        name += '_forget-greedy'
    if configs.model.use_prompt:
        name += '_prompt'
        if configs.model.use_scene_id:
            name += '-scene'
        if configs.train.strategy:
            name += f'-strategy{configs.train.strategy}'
    name += f'_sigma{configs.train.memory.sigma}'
    name += f'_temp{configs.train.memory.rank_temperature}'
    res_dir = os.path.join(configs.save_dir, configs.model.name, name)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    
    logger = SummaryWriter(os.path.join(configs.save_dir, configs.model.name, 'tf_logs'))
    
    # Load and save initial checkpoint
    print('Loading Initial Checkpoint: ', end = '')
    name = 'final_ckpt'
    name += f'_{configs.model.num_prompt_block}'
    initial_ckpt = os.path.join(model_dir, f'{name}.pth')
    assert os.path.exists(initial_ckpt), f'Initial Checkpoint at {initial_ckpt} should not be none'
    old_ckpt = torch.load(initial_ckpt)
    if inv:
        torch.save(old_ckpt, os.path.join(res_dir, f'env_0_inv.pth'))
    else:
        torch.save(old_ckpt, os.path.join(res_dir, f'env_0.pth'))
    print('Done')

    print('Loading Memory: ', end = '')
    # Make metric tracker, incremental memory
    metrics = IncrementalTracker()
    memory = Memory()
    
    # Update memory, metric tracker with env. 0
    model_frozen = model_factory(ckpt = old_ckpt, device = 'cuda')
    events = load_pickle(os.path.join(model_dir, f'events0_{configs.model.num_prompt_block}.pickle'))
    memory.update_memory(model_frozen, env_idx=0, forget_events=events)
    if not configs.debug:
        eval_stats = evaluate(model_frozen, env_idx=0)
        metrics.update(eval_stats, env_idx=0)
        metrics.update_rank(memory.num_keep, memory.entropies)
    print('Done')
    
    # Iterate over training steps
    for env_idx, env in enumerate(configs.train.incremental_environments):
        print(f'Training on environment # {env_idx + 1}')
        
        env_idx = env_idx + 1 # Start env_idx at 1

        # Make Trainer
        trainer = TrainerIncremental(logger, memory, env, old_ckpt, env_idx)

        # Train on this environment
        if configs.model.use_prompt and configs.train.strategy != 1:  # Two stage training: only QFormer -> except QFormer
            if configs.train.strategy == 2:
                # stage 1: train QFormer
                new_model = trainer.train(stage=1)
                if not configs.debug:
                    eval_stats = evaluate(new_model, env_idx)
                # stage 2: train except QFormer
                new_model = trainer.train(stage=2)
            else:  # strategy: 3 (train QFormer only)
                new_model = trainer.train(stage=1)
        else:  # Train one stage
            new_model = trainer.train(stage=-1)
        
        # Update Memory
        memory.update_memory(new_model, env_idx, tuples=trainer.dataloader.dataset.queries,
                             forget_events=trainer.get_forget_events())
        
        # Eval
        # if not configs.debug or env_idx == len(configs.train.incremental_environments):
        eval_stats = evaluate(new_model, env_idx)
        # Update Inc. Learning Stats
        metrics.update(eval_stats, env_idx)
        metrics.update_rank(memory.num_keep, memory.entropies)
        
        # Save model
        old_ckpt = new_model.state_dict()
        if inv:
            torch.save(old_ckpt, os.path.join(res_dir, f'env_{env_idx}_inv.pth'))
        else:
            torch.save(old_ckpt, os.path.join(res_dir, f'env_{env_idx}.pth'))
        
        # save forget events
        if configs.train.memory.use_greedy:
            events = trainer.get_forget_events()
            save_pickle(events, os.path.join(res_dir, f'events_{env_idx}.pickle'))

    # Print and save final results
    results_final = metrics.get_results()
    results_final.to_csv(os.path.join(res_dir, 'res.csv'))
    metrics.rank_df.to_csv(os.path.join(res_dir, 'rank.csv'))
    if not configs.debug:
        metrics.vis_grid_plot(filepath=os.path.join(res_dir, 'res.png'))
