import os
import shutil
import random


dataset_directory = 'data/small/'
train_directory = 'data/train_small/'
test_directory = 'data/test_small/'

random.seed(42)

if not os.path.exists(train_directory):
    os.makedirs(train_directory)
if not os.path.exists(test_directory):
    os.makedirs(test_directory)


for root, dirs, files in os.walk(dataset_directory):
    for dir in dirs:
        train_id_path = os.path.join(train_directory, dir)
        test_id_path = os.path.join(test_directory, dir)

        if not os.path.exists(train_id_path):
            os.makedirs(train_id_path)
        if not os.path.exists(test_id_path):
            os.makedirs(test_id_path)

        full_dir_path = os.path.join(root, dir)
        images = os.listdir(full_dir_path)

        random.shuffle(images)

        split_index = int(0.8 * len(images))

        train_files = images[:split_index]

        test_files = images[split_index:]

        for file in train_files:
            shutil.copy(os.path.join(full_dir_path, file), os.path.join(train_id_path, file))
        for file in test_files:
            shutil.copy(os.path.join(full_dir_path, file), os.path.join(test_id_path, file))

print("Successfully split data into train and test sets.")
