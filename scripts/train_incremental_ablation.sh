#!/bin/bash


_ROOT='<...>'

# config/protocols/4-step.yaml  or  config/protocols/4-step-hete.yaml
CONFIG_YAML='config/protocols/4-step-hete.yaml'
MODEL_NAME='MinkFPN_GeM'
BATCH_SIZE=100  # PatchAugNet=64

export PYTHONPATH=$PYTHONPATH:$_ROOT
cd $_ROOT

# train on the remaining scenes incrementally
# model name: PointNetVlad, MinkFPN_GeM, PatchAugNet
# memory selection: random or stochastic-greedy heuristic
# memory size: 0, 128, 256, 512
# use prompt: True (two stage), False (one stage)

################################### ablation of key modules ######################################
# # setting 1: random select - random forget
# echo "Train on remaining Dataset: no prompt, memory selection-random forget-random"
# python training/train_incremental.py \
#     --config $CONFIG_YAML \
#     model.name $MODEL_NAME \
#     train.memory.use_greedy False \
#     train.memory.num_pairs 256 \
#     model.use_prompt False \
#     train.batch_size_limit $BATCH_SIZE

# # setting 2: random select - greedy forget
# echo "Train on remaining Dataset: no prompt, memory selection-random forget-greedy"
# python training/train_incremental.py \
#     --config $CONFIG_YAML \
#     model.name $MODEL_NAME \
#     train.memory.use_greedy False \
#     train.memory.random_forget False \
#     train.memory.num_pairs 256 \
#     model.use_prompt False \
#     train.batch_size_limit $BATCH_SIZE

# # setting 3: greedy select (dist only) - random forget
# echo "Train on remaining Dataset: no prompt, memory selection-greedy forget-random"
# python training/train_incremental.py \
#     --config $CONFIG_YAML \
#     model.name $MODEL_NAME \
#     train.memory.use_greedy True \
#     train.memory.use_forget False \
#     train.memory.use_dist True \
#     train.memory.num_pairs 256 \
#     model.use_prompt False \
#     train.batch_size_limit $BATCH_SIZE

# # setting 4: greedy select (dist only) - greedy forget
# echo "Train on remaining Dataset: no prompt, memory selection-greedy forget-greedy"
# python training/train_incremental.py \
#     --config $CONFIG_YAML \
#     model.name $MODEL_NAME \
#     train.memory.use_greedy True \
#     train.memory.use_forget False \
#     train.memory.use_dist True \
#     train.memory.random_forget False \
#     train.memory.num_pairs 256 \
#     model.use_prompt False \
#     train.batch_size_limit $BATCH_SIZE


# # setting 5: greedy select (dist only, adaptive num of samples) - greedy forget
# echo "Train on remaining Dataset: no prompt, memory selection-greedy forget-greedy"
# python training/train_incremental.py \
#     --config $CONFIG_YAML \
#     model.name $MODEL_NAME \
#     train.memory.use_greedy True \
#     train.memory.use_forget False \
#     train.memory.use_dist True \
#     train.memory.random_forget False \
#     train.memory.num_pairs 256 \
#     train.memory.rank_temperature 4 \
#     model.use_prompt False \
#     train.batch_size_limit $BATCH_SIZE


# # setting 6: use prompt - greedy select (dist only, adaptive num of samples) - greedy forget--- ours!
# echo "Train on remaining Dataset: use prompt, memory selection-greedy forget-greedy"
# python training/train_incremental.py \
#     --config $CONFIG_YAML \
#     model.name $MODEL_NAME \
#     train.memory.use_greedy True \
#     train.memory.use_forget False \
#     train.memory.use_dist True \
#     train.memory.random_forget False \
#     train.memory.num_pairs 256 \
#     train.memory.rank_temperature 4 \
#     model.use_prompt True \
#     train.batch_size_limit $BATCH_SIZE


# # setting 7: (ours + scene id)
# echo "Train on remaining Dataset: use prompt, memory selection-greedy forget-greedy"
# python training/train_incremental.py \
#     --config $CONFIG_YAML \
#     model.name $MODEL_NAME \
#     train.memory.use_greedy True \
#     train.memory.use_forget False \
#     train.memory.use_dist True \
#     train.memory.random_forget False \
#     train.memory.num_pairs 256 \
#     train.memory.rank_temperature 4 \
#     model.use_prompt True \
#     model.use_scene_id True \
#     train.batch_size_limit $BATCH_SIZE


################################### threshold of rank_temperature ######################################
# # setting 5: greedy select (dist only, adaptive num of samples) - greedy forget
# echo "Train on remaining Dataset: no prompt, memory selection-greedy forget-greedy"
# python training/train_incremental.py \
#     --config $CONFIG_YAML \
#     model.name $MODEL_NAME \
#     train.memory.use_greedy True \
#     train.memory.use_forget False \
#     train.memory.use_dist True \
#     train.memory.random_forget False \
#     train.memory.num_pairs 256 \
#     train.memory.rank_temperature 0.5 \
#     model.use_prompt False \
#     train.batch_size_limit $BATCH_SIZE


# # setting 5: greedy select (dist only, adaptive num of samples) - greedy forget
# echo "Train on remaining Dataset: no prompt, memory selection-greedy forget-greedy"
# python training/train_incremental.py \
#     --config $CONFIG_YAML \
#     model.name $MODEL_NAME \
#     train.memory.use_greedy True \
#     train.memory.use_forget False \
#     train.memory.use_dist True \
#     train.memory.random_forget False \
#     train.memory.num_pairs 256 \
#     train.memory.rank_temperature 1 \
#     model.use_prompt False \
#     train.batch_size_limit $BATCH_SIZE


# # setting 5: greedy select (dist only, adaptive num of samples) - greedy forget
# echo "Train on remaining Dataset: no prompt, memory selection-greedy forget-greedy"
# python training/train_incremental.py \
#     --config $CONFIG_YAML \
#     model.name $MODEL_NAME \
#     train.memory.use_greedy True \
#     train.memory.use_forget False \
#     train.memory.use_dist True \
#     train.memory.random_forget False \
#     train.memory.num_pairs 256 \
#     train.memory.rank_temperature 2 \
#     model.use_prompt False \
#     train.batch_size_limit $BATCH_SIZE


# # setting 5: greedy select (dist only, adaptive num of samples) - greedy forget
# echo "Train on remaining Dataset: no prompt, memory selection-greedy forget-greedy"
# python training/train_incremental.py \
#     --config $CONFIG_YAML \
#     model.name $MODEL_NAME \
#     train.memory.use_greedy True \
#     train.memory.use_forget False \
#     train.memory.use_dist True \
#     train.memory.random_forget False \
#     train.memory.num_pairs 256 \
#     train.memory.rank_temperature 4 \
#     model.use_prompt False \
#     train.batch_size_limit $BATCH_SIZE


# # setting 5: greedy select (dist only, adaptive num of samples) - greedy forget
# echo "Train on remaining Dataset: no prompt, memory selection-greedy forget-greedy"
# python training/train_incremental.py \
#     --config $CONFIG_YAML \
#     model.name $MODEL_NAME \
#     train.memory.use_greedy True \
#     train.memory.use_forget False \
#     train.memory.use_dist True \
#     train.memory.random_forget False \
#     train.memory.num_pairs 256 \
#     train.memory.rank_temperature 8 \
#     model.use_prompt False \
#     train.batch_size_limit $BATCH_SIZE


# # setting 5: greedy select (dist only, adaptive num of samples) - greedy forget
# echo "Train on remaining Dataset: no prompt, memory selection-greedy forget-greedy"
# python training/train_incremental.py \
#     --config $CONFIG_YAML \
#     model.name $MODEL_NAME \
#     train.memory.use_greedy True \
#     train.memory.use_forget False \
#     train.memory.use_dist True \
#     train.memory.random_forget False \
#     train.memory.num_pairs 256 \
#     train.memory.rank_temperature 16 \
#     model.use_prompt False \
#     train.batch_size_limit $BATCH_SIZE


##################################### different training strategy ####################################
# one stage
echo "Train on remaining Dataset: use prompt, memory selection-greedy"
python training/train_incremental.py \
    --config $CONFIG_YAML \
    model.name $MODEL_NAME \
    train.memory.use_greedy True \
    train.memory.use_forget False \
    train.memory.use_dist True \
    train.memory.random_forget False \
    train.memory.num_pairs 256 \
    train.memory.rank_temperature 4 \
    train.strategy 1 \
    model.use_prompt True \
    train.batch_size_limit $BATCH_SIZE


# train prompt module only
echo "Train on remaining Dataset: use prompt, memory selection-greedy"
python training/train_incremental.py \
    --config $CONFIG_YAML \
    model.name $MODEL_NAME \
    train.memory.use_greedy True \
    train.memory.use_forget False \
    train.memory.use_dist True \
    train.memory.random_forget False \
    train.memory.num_pairs 256 \
    train.memory.rank_temperature 4 \
    train.strategy 3 \
    model.use_prompt True \
    train.batch_size_limit $BATCH_SIZE


################################### num of prompt block (optinal) ######################################
# # setting 1: num prompt block: 1
# echo "Train on remaining Dataset: use prompt, memory selection-greedy-256"
# python training/train_incremental.py \
#     --config $CONFIG_YAML \
#     model.name $MODEL_NAME \
#     train.memory.use_greedy True \
#     train.memory.num_pairs 256 \
#     model.use_prompt True \
#     model.num_prompt_block 1 \
#     train.batch_size_limit $BATCH_SIZE

# # setting 2: num prompt block: 2 (refer to QFormer)
# echo "Train on remaining Dataset: use prompt, memory selection-greedy-256"
# python training/train_incremental.py \
#     --config $CONFIG_YAML \
#     model.name $MODEL_NAME \
#     train.memory.use_greedy True \
#     train.memory.num_pairs 256 \
#     model.use_prompt True \
#     model.num_prompt_block 2 \
#     train.batch_size_limit $BATCH_SIZE

# # setting 3: num prompt block: 4
# echo "Train on remaining Dataset: use prompt, memory selection-greedy-256"
# python training/train_incremental.py \
#     --config $CONFIG_YAML \
#     model.name $MODEL_NAME \
#     train.memory.use_greedy True \
#     train.memory.num_pairs 256 \
#     model.use_prompt True \
#     model.num_prompt_block 4 \
#     train.batch_size_limit $BATCH_SIZE

# # setting 4: num prompt block: 8
# echo "Train on remaining Dataset: use prompt, memory selection-greedy-256"
# python training/train_incremental.py \
#     --config $CONFIG_YAML \
#     model.name $MODEL_NAME \
#     train.memory.use_greedy True \
#     train.memory.num_pairs 256 \
#     model.use_prompt True \
#     model.num_prompt_block 8 \
#     train.batch_size_limit $BATCH_SIZE


# ##################################### INV (optional) ####################################
# CONFIG_YAML='config/protocols/4-step-hete-inv.yaml'
# # setting 5 (ours)
# echo "Train on remaining Dataset: use prompt, memory selection-greedy"
# python training/train_incremental.py \
#     --config $CONFIG_YAML \
#     model.name $MODEL_NAME \
#     train.memory.use_greedy True \
#     train.memory.num_pairs 256 \
#     model.use_prompt True \
#     train.batch_size_limit $BATCH_SIZE