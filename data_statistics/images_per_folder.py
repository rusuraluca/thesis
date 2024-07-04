import os

dataset_directory = 'data/small/'

def count_folders(directory):
    small_folder_count = 0

    for root, dirs, files in os.walk(directory):
        for dir_name in dirs:

            full_dir_path = os.path.join(root, dir_name)

            num_images = len([name for name in os.listdir(full_dir_path) if os.path.isfile(os.path.join(full_dir_path, name))])

            if num_images < 20:
                small_folder_count += 1
                print(f"Folder {full_dir_path} with less than {num_images} images")

    return small_folder_count

small_folder_count = count_folders(dataset_directory)
print(f"Number of folders with less than 20 images: {small_folder_count}")
