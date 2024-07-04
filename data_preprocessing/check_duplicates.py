import os

def find_similar_folders(dataset_directory):
    prefix_dict = {}

    for folder in os.listdir(dataset_directory):
        if os.path.isdir(os.path.join(dataset_directory, folder)):
            prefix = folder.split('_', 1)[0]
            prefix = prefix.strip()

            if prefix in prefix_dict:
                prefix_dict[prefix].append(folder)
            else:
                prefix_dict[prefix] = [folder]

    similar_folders = {key: value for key, value in prefix_dict.items() if len(value) > 1}
    return similar_folders


dataset_directory = 'data/small/'

similar_folder_names = find_similar_folders(dataset_directory)

if similar_folder_names:
    print("Found folders with similar starting names:")
    for prefix, folders in similar_folder_names.items():
        print(f"Prefix '{prefix}' is used by folders: {', '.join(folders)}")
else:
    print("No folders with similar starting names found.")
