import torch
import pandas as pd
from PIL import Image
import numpy as np

# Possible Labels in csv file, Note: multi-labels are seperated by "|"
label_dict = {"Atelectasis": 0, "Consolidation":1, "Infiltration":2, "Pneumothorax":3, "Edema":4, 
            "Emphysema":5, "Fibrosis":6, "Effusion":7, "Pneumonia":8, "Pleural_thickening":9, 
            "Cardiomegaly":10, "Nodule":11, "Mass":12, "Hernia":13}

# Parse labels
def parse_csv_labels(df):
    labels = np.array(df['Finding Labels'])
    labels_tensor = torch.zeros(len(labels), len(label_dict))
    for ind_label in range(0, len(labels)): 
        labels_tensor[ind_label] = parse_ind(labels[ind_label])

    return labels_tensor

def parse_ind(ind_label):
    label_list = str(ind_label).split('|')
    label_tensor = torch.zeros(len(label_dict))
    for mark in label_list: 
        label_tensor[label_dict.get(mark)] = 1

    return label_tensor

# Custom Date set with parsing
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, csv, train, test, rootDir, transform):
        self.csv = csv
        self.train = train
        self.test = test
        self.rootDir = rootDir
        self.df = pd.read_csv(csv)
        self.all_image_names =  np.array(self.csv[:]['Image Index'])  # First column contains the image paths
        self.all_labels = parse_csv_labels(self.df)         # Second column is the labels, parsed and in tensor
        self.transform = transform                      # set transformer

        self.train_ratio = int(0.875 * len(self.csv))
        self.valid_ratio = len(self.csv) - self.train_ratio

        # set the training data images and labels
        if self.train == True:
            print(f"Number of training images: {self.train_ratio}")
            self.image_names = np.array(self.all_image_names[:self.train_ratio])
            self.labels = list(self.all_labels[:self.train_ratio])

        # set the validation data images and labels
        elif self.train == False and self.test == False:
            print(f"Number of validation images: {self.valid_ratio}")
            self.image_names = self.all_image_names[-self.valid_ratio:]
            self.labels = self.all_labels[-self.valid_ratio:]

        # set the test data images and labels from a seperate csv so no need to divide
        elif self.test == True and self.train == False:
            self.image_names = self.all_image_names
            self.labels = self.all_labels

    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, index):
        single_image_name = self.image_names[index]         # Get image name from the pandas df
        image = Image.open(self.rootDir+single_image_name)  # Open image

        # apply image transforms
        if self.transform is not None:
            image = self.transform(image)
        target = self.labels[index]
        
        return image, target