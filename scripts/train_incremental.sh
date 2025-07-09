#!/bin/bash


_ROOT='<...>'

# config/protocols/4-step.yaml  or  config/protocols/4-step-hete.yaml
CONFIG_YAML='config/protocols/4-step.yaml'
DATA_AUG=1


export PYTHONPATH=$PYTHONPATH:$_ROOT
cd $_ROOT

# train on the remaining scenes incrementally
# model name: PointNetVlad, MinkFPN_GeM, PatchAugNet


#########################################################################
MODEL_NAME='MinkFPN_GeM'
BATCH_SIZE=256
# setting 1
echo "Train on remaining Dataset: MinkFPN_GeM"
python training/train_continual.py \
    --config $CONFIG_YAML \
    model.name $MODEL_NAME \
    train.memory.num_pairs 256 \
    data.aug_mode $DATA_AUG \
    train.batch_size $BATCH_SIZE \
    train.batch_size_limit $BATCH_SIZE

#########################################################################
MODEL_NAME='PointNetVlad'
BATCH_SIZE=50
# setting 1
echo "Train on remaining Dataset: PointNetVlad"
python training/train_continual.py \
    --config $CONFIG_YAML \
    model.name $MODEL_NAME \
    train.memory.num_pairs 256 \
    data.aug_mode $DATA_AUG \
    train.batch_size $BATCH_SIZE \
    train.batch_size_limit $BATCH_SIZE

#########################################################################
MODEL_NAME='PatchAugNet'
BATCH_SIZE=60
# setting 1
echo "Train on remaining Dataset: PatchAugNet"
python training/train_continual.py \
    --config $CONFIG_YAML \
    model.name $MODEL_NAME \
    train.memory.num_pairs 256 \
    data.aug_mode $DATA_AUG \
    train.batch_size $BATCH_SIZE \
    train.batch_size_limit $BATCH_SIZE

# #########################################################################
# MODEL_NAME='MinkFPN_GeM'
# BATCH_SIZE=256
# CONFIG_YAML='config/protocols/4-step-hete.yaml'
# # setting 1
# echo "Train on remaining Dataset: MinkFPN_GeM"
# python training/train_continual.py \
#     --config $CONFIG_YAML \
#     model.name $MODEL_NAME \
#     train.memory.num_pairs 256 \
#     data.aug_mode $DATA_AUG \
#     train.batch_size $BATCH_SIZE \
#     train.batch_size_limit $BATCH_SIZE
