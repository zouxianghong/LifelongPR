#!/bin/bash


_ROOT='<...>'

# config/protocols/4-step.yaml  or  config/protocols/4-step-hete.yaml
CONFIG_YAML='config/protocols/4-step-hete.yaml'


export PYTHONPATH=$PYTHONPATH:$_ROOT
cd $_ROOT

# train on the remaining scenes incrementally
# model name: PointNetVlad, MinkFPN_GeM, PatchAugNet
# memory selection: random or stochastic-greedy heuristic
# memory size: 0, 128, 256, 512
# use prompt: True (two stage), False (one stage)


# #########################################################################
# MODEL_NAME='MinkFPN_GeM'
# BATCH_SIZE=120
# # setting 1: use prompt - greedy select (dist only, adaptive num of samples) - greedy forget--- ours!
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

# # setting 2: Fine Tune
# echo "Train on remaining Dataset: no prompt, no memory"
# python training/train_incremental.py \
#     --config $CONFIG_YAML \
#     model.name $MODEL_NAME \
#     train.memory.use_greedy False \
#     train.memory.num_pairs 0 \
#     model.use_prompt False \
#     train.batch_size_limit $BATCH_SIZE


# #########################################################################
# MODEL_NAME='PointNetVlad'
# BATCH_SIZE=40
# # setting 1: use prompt - greedy select (dist only, adaptive num of samples) - greedy forget--- ours!
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

# # setting 2: Fine Tune
# echo "Train on remaining Dataset: no prompt, no memory"
# python training/train_incremental.py \
#     --config $CONFIG_YAML \
#     model.name $MODEL_NAME \
#     train.memory.use_greedy False \
#     train.memory.num_pairs 0 \
#     model.use_prompt False \
#     train.batch_size_limit $BATCH_SIZE


#########################################################################
MODEL_NAME='PatchAugNet'
BATCH_SIZE=80
# setting 1: use prompt - greedy select (dist only, adaptive num of samples) - greedy forget--- ours!
echo "Train on remaining Dataset: use prompt, memory selection-greedy forget-greedy"
python training/train_incremental.py \
    --config $CONFIG_YAML \
    model.name $MODEL_NAME \
    train.memory.use_greedy True \
    train.memory.use_forget False \
    train.memory.use_dist True \
    train.memory.random_forget False \
    train.memory.num_pairs 256 \
    train.memory.rank_temperature 4 \
    model.use_prompt True \
    train.batch_size_limit $BATCH_SIZE

# # setting 2: Fine Tune
# echo "Train on remaining Dataset: no prompt, no memory"
# python training/train_incremental.py \
#     --config $CONFIG_YAML \
#     model.name $MODEL_NAME \
#     train.memory.use_greedy False \
#     train.memory.num_pairs 0 \
#     model.use_prompt False \
#     train.batch_size_limit $BATCH_SIZE
