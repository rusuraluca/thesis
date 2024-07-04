import os

dataset_directory = 'data/small/'

age_counts = {age: 0 for age in range(67)}

total = 0
for root, dirs, files in os.walk(dataset_directory):
    for file in files:
        if file.endswith(".jpg"):
            try:
                age_part = file.split('_')[-1]
                age = int(age_part.split('.')[0])
                if 0 <= age <= 65:
                    age_counts[age] += 1
                    total += 1
                else :
                    age_counts[66] += 1
                    total += 1
            except ValueError:
                print(f"Could not extract age from {file}")

for age, count in sorted(age_counts.items()):
    print(f"Age {age}: {count} images")

print(f"Total {total} images")
