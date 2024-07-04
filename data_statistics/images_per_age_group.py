import os

dataset_directory = 'data/small/'

age_groups = {
    "0-25": 0,
    "26-55": 0,
    "55+": 0,
}

def age_to_group(age):
    if age <= 25:
        return "0-25"
    elif age <= 55:
        return "26-55"
    else:
        return "55+"

total = 0
for root, dirs, files in os.walk(dataset_directory):
    for file in files:
        if file.endswith(".jpg"):
            try:
                age_part = file.split('_')[-1]
                age = int(age_part.split('.')[0])
                group = age_to_group(age)
                age_groups[group] += 1
                total += 1
            except ValueError:
                print(f"Could not extract age from {file}")

for group, count in age_groups.items():
    print(f"Age group {group}: {count} images")

print(f"Total {total} images")
