import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from aifr.utils.image_loader import ImageLoader


class AnalysisStrategy(ABC):
    """
    Abstract base class for analysis strategies.
    """
    @abstractmethod
    def analyze(self, labels, ages, genders):
        """
        Abstract method to perform analysis.

        Args:
            labels (list): List of labels.
            ages (list): List of ages.
            genders (list): List of genders.
        """
        pass


class AgeGenderDistributionAnalysis(AnalysisStrategy):
    """
    Concrete strategy class for age-gender distribution analysis.
    """
    def analyze(self, labels, ages, genders):
        """
        Perform age-gender distribution analysis and plot the results.

        Args:
            labels (list): List of labels.
            ages (list): List of ages.
            genders (list): List of genders.
        """
        age_gender_groups = {
            0: {'Label': '0-19', 'Male': 0, 'Female': 0},
            1: {'Label': '20-25', 'Male': 0, 'Female': 0},
            2: {'Label': '26-35', 'Male': 0, 'Female': 0},
            3: {'Label': '36-45', 'Male': 0, 'Female': 0},
            4: {'Label': '46-55', 'Male': 0, 'Female': 0},
            5: {'Label': '56-65', 'Male': 0, 'Female': 0},
            6: {'Label': '>=65', 'Male': 0, 'Female': 0}
        }

        for age, gender in zip(ages, genders):
            if gender == 0:
                age_gender_groups[age]['Male'] += 1
            else:
                age_gender_groups[age]['Female'] += 1

        fig, ax = plt.subplots(figsize=(10, 6))
        age_groups = [group['Label'] for group in age_gender_groups.values()]
        num_age_groups = len(age_groups)
        bar_width = 0.35
        index = np.arange(num_age_groups)

        male_counts = [group['Male'] for group in age_gender_groups.values()]
        female_counts = [group['Female'] for group in age_gender_groups.values()]

        ax.bar(index, male_counts, bar_width, label='Male', color='dodgerblue')
        ax.bar(index + bar_width, female_counts, bar_width, label='Female', color='mediumpurple')

        ax.set_xlabel('Age Group')
        ax.set_ylabel('Image Count')
        ax.set_title('Age-Gender Distribution')
        ax.set_xticks(index + bar_width / 2)
        ax.set_xticklabels(age_groups)
        ax.legend()

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()


class GenderDistributionAnalysis(AnalysisStrategy):
    """
    Concrete strategy class for gender distribution analysis.
    """
    def analyze(self, labels, ages, genders):
        """
        Perform gender distribution analysis and plot the results.

        Args:
            labels (list): List of labels.
            ages (list): List of ages.
            genders (list): List of genders.
        """
        male_count = sum(g == 0 for g in genders)
        female_count = sum(g == 1 for g in genders)
        total_count = male_count + female_count

        if total_count == 0:
            print("No data available for analysis.")
            return

        male_percentage = (male_count / total_count) * 100
        female_percentage = (female_count / total_count) * 100

        labels = 'Male', 'Female'
        sizes = [male_percentage, female_percentage]
        colors = ['dodgerblue', 'mediumpurple']
        explode = (0.1, 0)

        plt.figure(figsize=(6, 6))
        plt.pie(sizes, explode=explode, labels=labels, colors=colors,
                autopct='%1.1f%%', startangle=140)
        plt.title(f'Gender Distribution: Male ({male_count}) vs. Female ({female_count})')
        plt.axis('equal')
        plt.show()


class DatasetAnalysisContext:
    """
    Context class for performing dataset analysis using different strategies.

    Attributes:
        strategy (AnalysisStrategy): The strategy to use for analysis.
        dataset_path (str): Path to the dataset.
        pattern (str): Pattern to match in filenames.
        position_age (int): Position of the age information in the filename.
        position_gender (int): Position of the gender information in the filename.
        cutoffs (list): List of age cutoffs for grouping.
        transforms (transforms.Compose): Transformations to apply to the images.
    """
    def __init__(self,
                strategy: AnalysisStrategy,
                dataset_path: str,
                pattern=r'_|\.',
                position_age=1,
                position_gender=1,
                cutoffs=None):
        """
        Initialize the context with a strategy and dataset information.

        Args:
            strategy (AnalysisStrategy): The strategy to use for analysis.
            dataset_path (str): Path to the dataset.
            pattern (str, optional): Pattern to match in filenames. Default is r'_|\.'.
            position_age (int, optional): Position of the age information in the filename. Default is 1.
            position_gender (int, optional): Position of the gender information in the filename. Default is 1.
            cutoffs (list, optional): List of age cutoffs for grouping. Default is None.
        """
        if cutoffs is None:
            cutoffs = [19, 25, 35, 45, 55, 65]
        self.strategy = strategy
        self.dataset_path = dataset_path
        self.pattern = pattern
        self.position_age = position_age
        self.position_gender = position_gender
        self.cutoffs = cutoffs
        self.transforms = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.CenterCrop(160),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def set_strategy(self, strategy: AnalysisStrategy):
        """
        Set a new analysis strategy.

        Args:
            strategy (AnalysisStrategy): The new strategy to use for analysis.
        """
        self.strategy = strategy

    def perform_analysis(self):
        """
        Perform analysis using the current strategy.

        Loads the dataset, processes the data, and applies the analysis strategy.
        """
        dataset = ImageLoader(root=self.dataset_path, pattern=self.pattern, position_age=self.position_age, position_gender=self.position_gender, cutoffs=self.cutoffs, transform=self.transforms)
        loader = DataLoader(dataset, batch_size=512, shuffle=False)

        ages = []
        genders = []
        labels = []
        for _, label, age_group, gender_group in loader:
            ages.extend(age_group.numpy())
            genders.extend(gender_group.numpy())
            labels.extend(label.numpy())
        self.strategy.analyze(labels, ages, genders)


if __name__ == "__main__":
    """
    Main function to perform dataset analysis with different strategies.
    """
    path = './data/big'
    context = DatasetAnalysisContext(AgeGenderDistributionAnalysis(), path)
    context.perform_analysis()
    context.set_strategy(GenderDistributionAnalysis())
    context.perform_analysis()

    path = './data/small'
    context = DatasetAnalysisContext(AgeGenderDistributionAnalysis(), path)
    context.perform_analysis()
    context.set_strategy(GenderDistributionAnalysis())
    context.perform_analysis()

    path = './data/fgnet'
    context = DatasetAnalysisContext(AgeGenderDistributionAnalysis(), path)
    context.perform_analysis()
    context.set_strategy(GenderDistributionAnalysis())
    context.perform_analysis()
