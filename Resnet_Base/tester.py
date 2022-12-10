
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

epoch = 50
lr = 0.001
batch_size = 64
pretrain = False
optimizer = 'adam'
lr_sched = 'cosine'
verbose = False
thresholds = [0.4, 0.5, 0.6]


for thresh in thresholds:
    TEST_NAME = f'chest_xray_e{epoch}_b{batch_size}_t{thresh}'
    print("Running Expirement: ",TEST_NAME)

    command = (
        f'python main.py --exp_name {TEST_NAME} --epochs {epoch} --batch_size {batch_size} ' 
        f'--lr {lr} --optimizer {optimizer} --lr_sched {lr_sched}  --thresh {thresh}' )
    os.system(command)