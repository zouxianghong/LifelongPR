#!/bin/bash


_ROOT='<...>'

# config yaml file: 'config/protocols/4-step.yaml' or 'config/protocols/4-step-hete.yaml'
CONFIG_YAML='config/protocols/4-step.yaml'
DATA_AUG=1

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONPATH=$PYTHONPATH:$_ROOT
cd $_ROOT

# train on the first scene
# model name: PointNetVlad, MinkFPN_GeM, PatchAugNet


#########################################################################
MODEL_NAME='MinkFPN_GeM'
BATCH_SIZE=256
# don't use mutual info (mutual info max)
echo "Train on 1st Dataset: MinkFPN_GeM"
python training/train.py \
    --config $CONFIG_YAML \
    model.name $MODEL_NAME \
    data.aug_mode $DATA_AUG \
    train.batch_size_limit $BATCH_SIZE

#########################################################################
MODEL_NAME='PointNetVlad'
BATCH_SIZE=100
# don't use mutual info (mutual info max)
echo "Train on 1st Dataset: PointNetVlad"
python training/train.py \
    --config $CONFIG_YAML \
    model.name $MODEL_NAME \
    data.aug_mode $DATA_AUG \
    train.batch_size_limit $BATCH_SIZE

#########################################################################
MODEL_NAME='PatchAugNet'
BATCH_SIZE=160
# don't use mutual info (mutual info max)
echo "Train on 1st Dataset: PatchAugNet"
python training/train.py \
    --config $CONFIG_YAML \
    model.name $MODEL_NAME \
    data.aug_mode $DATA_AUG \
    train.batch_size_limit $BATCH_SIZE
