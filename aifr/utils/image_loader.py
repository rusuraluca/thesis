import re
from os.path import basename
from torchvision.datasets import ImageFolder


age_cutoffs = [25, 55]
genders = [0, 1]
folder_pattern = r'_|\.'

class ImageLoader(ImageFolder):
    """
    Custom dataset loader that extends torchvision's ImageFolder.

    This class loads images and extracts additional metadata (age and gender) from the image paths.

    Attributes:
        pattern (str): The pattern used to split the image paths.
        position_age (int): The position of the age information in the split path components.
        position_gender (int): The position of the gender information in the split path components.
        cutoffs (list): The age cutoffs for categorizing age groups.
    """
    def __init__(self,
                root,
                pattern=folder_pattern,
                position_age=1,
                position_gender=1,
                cutoffs=age_cutoffs,
                transform=None):
        """
        Initializes the ImageLoader with the specified parameters.

        Args:
            root (str): Root directory path.
            pattern (str, optional): Pattern to split the path components. Default is folder_pattern.
            position_age (int, optional): Position of the age information in the split path components. Default is 1.
            position_gender (int, optional): Position of the gender information in the split path components. Default is 1.
            cutoffs (list, optional): Age cutoffs for categorizing age groups. Default is age_cutoffs.
            transform (callable, optional): A function/transform that takes in an PIL image and returns a transformed version. Default is None.
        """
        super().__init__(root, transform=transform)
        self.pattern = pattern
        self.position_age = position_age
        self.position_gender = position_gender
        self.cutoffs = cutoffs

    def __getitem__(self, index):
        """
        Initializes the ImageLoader with the specified parameters.

        Args:
            root (str): Root directory path.
            pattern (str, optional): Pattern to split the path components. Default is folder_pattern.
            position_age (int, optional): Position of the age information in the split path components. Default is 1.
            position_gender (int, optional): Position of the gender information in the split path components. Default is 1.
            cutoffs (list, optional): Age cutoffs for categorizing age groups. Default is age_cutoffs.
            transform (callable, optional): A function/transform that takes in an PIL image and returns a transformed version. Default is None.
        """
        img, label = super().__getitem__(index)
        path = self.imgs[index][0]
        age = self.path2age(path, self.pattern, self.position_age)
        age_group = self.find_age_group(age)
        gender = self.path2gender(path, self.pattern, self.position_gender)
        gender_group = self.find_gender_group(gender)
        return img, label, age_group, gender_group

    @staticmethod
    def path2age(path, pat, pos):
        """
        Extracts the age from the image path.

        Args:
            path (str): The path of the image file.
            pat (str): The pattern used to split the path components.
            pos (int): The position of the age information in the split path components.

        Returns:
            int: The age extracted from the path.
        """
        return int(re.split(pat, basename(path))[pos])

    @staticmethod
    def path2gender(path, pat, pos):
        """
        Extracts the gender from the image path.

        Args:
            path (str): The path of the image file.
            pat (str): The pattern used to split the path components.
            pos (int): The position of the gender information in the split path components.

        Returns:
            int: 0 if the gender is male, 1 if the gender is female.
        """
        components = path.split('/')
        name_gender_dir = components[-2]
        gender_str = re.split(pat, name_gender_dir)[pos]
        return 0 if gender_str.lower().startswith('m') else 1

    def find_age_group(self, age):
        """
        Determines the age group for a given age based on cutoffs.

        Args:
            age (int): The age to be categorized.

        Returns:
            int: The index of the age group.
        """
        age_group = next((i for i, cutoff in enumerate(self.cutoffs) if age <= cutoff), len(self.cutoffs))
        return age_group

    @staticmethod
    def find_gender_group(gender):
        """
        Determines the gender group.

        Args:
            gender (int): The gender to be categorized.

        Returns:
            int: The gender group (same as input gender).
        """
        return gender
