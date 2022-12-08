# performs a hyperparameter sweep with many tests
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import pickle


BASE_TEST_NAME = f'base_resnet50'

epochs = [50]
lrs = [0.001, 0.00001]
batch_size = 128
pretrain = False
optimizers = ['adam']
lr_scheds = ['cosine']
verbose = [False]

results = {}

for epoch in epochs:
    for lr in lrs:
        for optimizer in optimizers:
            for lr_sched in lr_scheds:
                TEST_NAME = TEST_NAME + f'_e{epoch}_lr{lr}_pF_O{optimizer}_S{lr_sched}'

                command = (
                    f'python main.py --exp_name {TEST_NAME} --epochs {epoch} --lr {lr} '
                    f'--batch_size {batch_size} --pretrn {pretrain} --optimizer {optimizer}'
                    f'--lr_sched {lr_sched} --verbose {verbose}' )
