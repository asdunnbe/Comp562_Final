import os
import shutil

# directories
source_folder = "/Users/User/Github Repositeries/Comp562_Final/Chest_XRay/"
train_folder = "/Users/User/Github Repositeries/Comp562_Final/Chest_XRay/img_train"
test_folder = "/Users/User/Github Repositeries/Comp562_Final/Chest_XRay/img_test"


# test images
with open(source_folder + 'Text_Documents/test_list.txt', 'r') as f:
    test_list = [line.strip() for line in f]

# train images
with open(source_folder + 'Text_Documents/train_val_list.txt', 'r') as f:
    train_list = [line.strip() for line in f]


# fetch all files
for file_name in os.listdir(source_folder):
    # construct full file path
    if "images_" in file_name :
        source = source_folder + file_name + '/images'
        print(source)
        for img in os.listdir(source):
            if (img in test_list): destination_folder = test_folder
            if (img in train_list): destination_folder = train_folder

            # move files
            shutil.move(source + '/' + img, destination_folder + '/' + img)

print("done")