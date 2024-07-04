import os


dataset_directory = 'data/small/'


def count_gender_images(directory):
    gender_count = {'m': 0, 'f': 0}
    image_count = {'m': 0, 'f': 0}

    for root, dirs, files in os.walk(directory):
        for dir_name in dirs:
            if '_' in dir_name:
                identity, gender = dir_name.split('_')
                if gender in gender_count:
                    gender_count[gender] += 1
                    full_dir_path = os.path.join(root, dir_name)
                    num_images = len([name for name in os.listdir(full_dir_path) if os.path.isfile(os.path.join(full_dir_path, name))])
                    image_count[gender] += num_images

    return gender_count, image_count

gender_count, image_count = count_gender_images(dataset_directory)
print("Gender Identity Count:", gender_count)
print("Image Count per Gender:", image_count)
