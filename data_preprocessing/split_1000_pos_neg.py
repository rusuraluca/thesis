import os
import shutil
import random

def create_dataset_pairs(dataset_path, output_path, age_difference=15, num_pairs_per_gender=500):
    persons = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    male_persons = [p for p in persons if p.endswith('_m')]
    female_persons = [p for p in persons if p.endswith('_f')]

    positive_path = os.path.join(output_path, 'positive')
    negative_path = os.path.join(output_path, 'negative')
    os.makedirs(positive_path, exist_ok=True)
    os.makedirs(negative_path, exist_ok=True)

    def create_pairs(persons, path, is_positive=True):
        num_pairs = 0
        while num_pairs < num_pairs_per_gender * 2:
            person1 = random.choice(persons)
            person2 = person1 if is_positive else random.choice(persons)
            while person1 == person2:
                person2 = random.choice(persons)

            img_files1 = sorted(os.listdir(os.path.join(dataset_path, person1)))
            img_files2 = sorted(os.listdir(os.path.join(dataset_path, person2)))

            for img1 in img_files1:
                for img2 in img_files2:
                    age1 = int(img1.split('_')[0])
                    age2 = int(img2.split('_')[0])
                    if is_positive or abs(age1 - age2) >= age_difference:
                        pair_folder = os.path.join(path, f'{person1}_{img1}_with_{person2}_{img2}')
                        os.makedirs(pair_folder, exist_ok=True)
                        shutil.copy(os.path.join(dataset_path, person1, img1), os.path.join(pair_folder, img1))
                        shutil.copy(os.path.join(dataset_path, person2, img2), os.path.join(pair_folder, img2))
                        num_pairs += 1
                        if num_pairs >= num_pairs_per_gender * 2:
                            return

    create_pairs(persons, positive_path, is_positive=True)
    create_pairs(male_persons + female_persons, negative_path, is_positive=False)

dataset_path = 'data/fgnet/'
output_path = 'data/fgnet_split/'
create_dataset_pairs(dataset_path, output_path)
