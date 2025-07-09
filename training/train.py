import os 
import torch 
import argparse 
import random 
import numpy as np

from torchpack.utils.config import configs 
from misc.utils import save_pickle

from eval.evaluate import evaluate
from trainer import Trainer

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
    configs.train.memory.num_pairs = 0
    configs.model.use_prompt = False
    configs.model.use_scene_id = False

    if isinstance(configs.train.optimizer.scheduler_milestones, str):
        configs.train.optimizer.scheduler_milestones = [int(x) for x in configs.train.optimizer.scheduler_milestones.split(',')] # Allow for passing multiple drop epochs to scheduler
    print(configs)

    # Make save directory and logger
    if not os.path.exists(os.path.join(configs.save_dir, configs.model.name, 'models')):
        os.makedirs(os.path.join(configs.save_dir, configs.model.name, 'models'))
    logger = SummaryWriter(os.path.join(configs.save_dir, configs.model.name, 'tf_logs'))
    
    # Train model
    trainer = Trainer(logger, configs.train.initial_environment)
    trained_model = trainer.train()
    
    # save forget events
    events = trainer.get_forget_events()
    save_pickle(events, os.path.join(configs.save_dir, configs.model.name, 'models', f'events0_{configs.model.num_prompt_block}.pickle'))

    # Evaluate
    eval_stats = evaluate(trained_model, -1)
    
    # Save model
    name = 'final_ckpt'
    name += f'_{configs.model.num_prompt_block}'
    torch.save(trained_model.state_dict(), os.path.join(configs.save_dir, configs.model.name, 'models', f'{name}.pth'))
