#!/usr/bin/bash

export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3

python octaflow/train_oca2odt_save_all.py configs_release/anesthetized_dataset/anesthetized_arguments_train_lr_2e-4_save_all_fold_0.txt
