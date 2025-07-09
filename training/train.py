import os 
import torch 
import argparse 
import random 
import numpy as np 
import itertools 

from torchpack.utils.config import configs 
from datasets.memory import Memory

from eval.evaluate import evaluate 
from eval.metrics import IncrementalTracker 
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
    parser.add_argument('--model', type = str, required = False, help = 'Model name: logg3d, PointNetVlad, MinkFPN_GeM, PatchAugNet')
    
    args, opts = parser.parse_known_args()
    configs.load(args.config, recursive = True)
    configs.update(opts)
    if args.model is not None:
        configs.model.name = args.model

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

    # Evaluate 
    eval_stats = evaluate(trained_model, -1)
    # TODO Improve printing on eval part of train loop

    # Save model
    ckpt = trained_model.state_dict()
    torch.save(ckpt, os.path.join(configs.save_dir, configs.model.name, 'models', 'final_ckpt.pth'))
