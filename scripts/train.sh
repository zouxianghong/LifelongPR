#!/bin/bash


_ROOT='<...>'

# config/protocols/4-step.yaml  or  config/protocols/4-step-hete.yaml
CONFIG_YAML='config/protocols/4-step-hete.yaml'

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONPATH=$PYTHONPATH:$_ROOT
cd $_ROOT

# train on the first scene
# model name: PointNetVlad, MinkFPN_GeM, PatchAugNet


#########################################################################
MODEL_NAME='MinkFPN_GeM'
BATCH_SIZE=256
echo "Train on 1st Dataset"
python training/train.py \
    --config $CONFIG_YAML \
    model.name $MODEL_NAME \
    train.batch_size_limit $BATCH_SIZE


#########################################################################
MODEL_NAME='PointNetVlad'
BATCH_SIZE=100
echo "Train on 1st Dataset"
python training/train.py \
    --config $CONFIG_YAML \
    model.name $MODEL_NAME \
    train.batch_size_limit $BATCH_SIZE


#########################################################################
MODEL_NAME='PatchAugNet'
BATCH_SIZE=160
echo "Train on 1st Dataset"
python training/train.py \
    --config $CONFIG_YAML \
    model.name $MODEL_NAME \
    train.batch_size_limit $BATCH_SIZE
 