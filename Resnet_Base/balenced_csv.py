from turtle import done
import pandas as pd
import numpy as np
import random

# directory
metadata_train = "Users/User/Github Repositeries/Comp562_Final/Chest_XRay/Data_train_sing.csv"
metadata_test = "Users/User/Github Repositeries/Comp562_Final/Chest_XRay/Data_test_sing.csv"

test_csv = "Users/User/Github Repositeries/Comp562_Final/Chest_XRay/Data_test_bal.csv"
train_csv = "Users/User/Github Repositeries/Comp562_Final/Chest_XRay/Data_train_bal.csv"

metadata = [metadata_train, metadata_test]
save_csv = [train_csv, test_csv]

# 7k, 2.5k
for type in [0,1]:
    # organize into counts
    df = pd.read_csv(metadata[type])
    pathology = np.array(df['Finding Labels'])
    names = np.array(df['Image Index'])
    pathology = pathology.tolist()

    no_findings = []

    for i in range(0,len(pathology)):
        entry = pathology[i]
        if entry == 'No Finding': 
            no_findings.append(names[i])

    if metadata == metadata_train: select = 7000
    else: select = 2500

    rand_nf = random.choices(no_findings, k=select)

    not_list = [x for x in no_findings if x not in rand_nf]
    index_list = [x for x in names if x not in not_list]

    # append image row to correct csv
    for (img_index), group in df.groupby(['Image Index']):
        if img_index in index_list: 
            group.to_csv(save_csv[type], mode='a', index=False, header=False)

    headerList = list(df.columns)
    file = pd.read_csv(save_csv[type])
    file.to_csv(save_csv[type], header=headerList, index=False)

print('done')
