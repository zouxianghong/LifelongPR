import os 
import torch 
import argparse 
import random 
import numpy as np 
import itertools 

from torchpack.utils.config import configs 
from datasets.memory import Memory
from models.model_factory import model_factory

from eval.evaluate import evaluate 
from eval.metrics import IncrementalTracker 
from trainer_continual import TrainerIncremental

from torch.utils.tensorboard import SummaryWriter



if __name__ == '__main__':

    # Repeatability 
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    # Get args and configs 
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type = str, default = "config/protocols/4-step.yaml", required = False, help = 'Path to configuration YAML file')
    parser.add_argument('--model', type = str, required = False, help = 'Model name: logg3d, PointNetVlad, MinkFPN_GeM, PatchAugNet')

    args, opts = parser.parse_known_args()
    configs.load(args.config, recursive = True)
    configs.update(opts)
    if args.model is not None:
        configs.model.name = args.model
    
    if isinstance(configs.train.optimizer.scheduler_milestones, str):
        # decrease the learning rate at certain epochs
        configs.train.optimizer.scheduler_milestones = [int(x) for x in configs.train.optimizer.scheduler_milestones.split(',')] # Allow for passing multiple drop epochs to scheduler
    print(configs)

    # Make save directory and logger
    if not os.path.exists(os.path.join(configs.save_dir, configs.model.name, 'models')):
        os.makedirs(os.path.join(configs.save_dir, configs.model.name, 'models'))
    logger = SummaryWriter(os.path.join(configs.save_dir, configs.model.name, 'tf_logs'))

    # Load and save initial checkpoint
    print('Loading Initial Checkpoint: ', end = '')
    initial_ckpt = os.path.join(configs.save_dir, configs.model.name, "models", 'final_ckpt.pth')
    assert os.path.exists(initial_ckpt), f'Initial Checkpoint at {initial_ckpt} should not be none'
    # the trained model in the last step
    old_ckpt = torch.load(initial_ckpt)
    torch.save(old_ckpt, os.path.join(configs.save_dir, configs.model.name, 'models', f'env_0.pth'))
    print('Done')

    print('Loading Memory: ', end = '')
    # Make metric tracker, incremental memory
    metrics = IncrementalTracker()
    memory = Memory()

    # Update memory, metric tracker with env. 0
    '''
        configs.train.initial_environment: pickles/Oxford/Oxford_train_queries.pickle
        env_idx: 0
        configs.train.incremental_environments: pickles/combined_train_queries.pickle or other incremental_pickle
    '''
    memory.update_memory(configs.train.initial_environment, env_idx = 0)
    model_frozen = model_factory(ckpt = old_ckpt, device = 'cuda')
    eval_stats = evaluate(model_frozen, env_idx=0)
    metrics.update(eval_stats, env_idx = 0)
    print('Done')

    # Iterate over training steps
    old_env = initial_ckpt
    for env_idx, env in enumerate(configs.train.incremental_environments):
        print(f'Training on environment # {env_idx + 1}')
        env_idx = env_idx + 1 # Start env_idx at 1

        # Make Trainer
        trainer = TrainerIncremental(logger, memory, old_env, env, old_ckpt, env_idx)
        torch.cuda.empty_cache() 
        old_env = env # For EWC 

        # Train on this environment
        new_model = trainer.train()
        if not configs.debug or env_idx == len(configs.train.incremental_environments):
            eval_stats = evaluate(new_model, env_idx)
            # Update Inc. Learning Stats
            metrics.update(eval_stats, env_idx)

        # Save model
        old_ckpt = new_model.state_dict()
        torch.save(old_ckpt, os.path.join(configs.save_dir, configs.model.name, 'models', f'env_{env_idx}.pth'))

        # Update Memory
        memory.update_memory(env, env_idx)

    # Print and save final results
    results_final = metrics.get_results()
    results_final.to_csv(os.path.join(configs.save_dir, configs.model.name, 'results.csv'))
    metrics.vis_grid_plot(filepath=os.path.join(configs.save_dir, configs.model.name, 'results.png'))
