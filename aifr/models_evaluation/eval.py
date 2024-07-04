import os
import sys
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from config import EVAL
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)
from aifr.utils.model_handler import ModelHandler


class Evaluation:
    """
    Class for evaluating a model on positive and negative image pairs.
    """
    def __init__(self):
        """
        Initializes the Evaluation class by loading the configuration and model,
        then starts the evaluation process.
        """
        self.config = EVAL.instance()

        self.model = ModelHandler().get_model(self.config['model_name'], self.config['dataset'], self.config['margin_loss'])

        if self.config["model_weights"]:
            self.model.load_state_dict(torch.load(self.config["model_weights"]))
            print(f'Loaded weights from {self.config["model_weights"]}')

        self.model.eval()

        self.eval_dataset = self.config["eval_dataset"]

        accuracy_pos, mean_similarity = self.evaluate(os.path.join(self.eval_dataset, 'positive'), threshold=0.5)
        print(f'Evaluation positive pairs accuracy: {accuracy_pos}')
        print(f'Evaluation positive pairs mean cosine similarities: {mean_similarity}')

        accuracy_neg, mean_similarity = self.evaluate(os.path.join(self.eval_dataset, 'negative'), threshold=0.5)
        print(f'Evaluation negative pairs accuracy: {accuracy_neg}')
        print(f'Evaluation negative pairs mean cosine similarities: {mean_similarity}')

        accuracy = (accuracy_pos + accuracy_neg) / 2
        print(f'Evaluation total accuracy: {accuracy}')

    def evaluate(self, directory, threshold=0.5):
        """
        Evaluates the model on image pairs in the specified directory.

        Args:
            directory (str): Path to the directory containing image pairs.
            threshold (float, optional): The threshold for determining if two images are of the same person. Default is 0.5.

        Returns:
            tuple: A tuple containing the accuracy and mean cosine similarity of the evaluated pairs.
        """
        def load_and_transform_image(image_path, transform):
            """
            Loads and transforms an image from the specified path.

            Args:
                image_path (str): Path to the image file.
                transform (transforms.Compose): The transformations to apply to the image.

            Returns:
                torch.Tensor: The transformed image tensor.
            """
            image = Image.open(image_path).convert('RGB')
            image = transform(image)
            # add an extra dimension of size 1 at the 0th position to the image tensor as a batch dimension, resulting in a shape of [1, CHANNELS=3, HEIGHT, WIDTH]
            return image.unsqueeze(0)

        transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.CenterCrop(160),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        transform_flipped = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.CenterCrop(160),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        similarities = []
        correct_predictions = 0
        total_predictions = 0

        for root, dirs, files in os.walk(directory):
            if len(files) == 2:
                file1, file2 = [os.path.join(root, f) for f in files]
                image1 = load_and_transform_image(file1, transform)
                image2 = load_and_transform_image(file2, transform)
                image1_flipped = load_and_transform_image(file1, transform_flipped)
                image2_flipped = load_and_transform_image(file2, transform_flipped)

                with torch.no_grad():
                    features1 = self.model(image1, return_embeddings=True)
                    features2 = self.model(image2, return_embeddings=True)
                    features1_flipped = self.model(image1_flipped, return_embeddings=True)
                    features2_flipped = self.model(image2_flipped, return_embeddings=True)
                    similarity = torch.cosine_similarity(features1 + features1_flipped, features2 + features2_flipped)

                similarities.append(similarity.item())

                is_same_person = similarity >= threshold
                is_positive_folder = 'positive' in root
                if is_same_person == is_positive_folder:
                    correct_predictions += 1
                total_predictions += 1

        accuracy = correct_predictions / total_predictions
        mean_similarity = np.mean(similarities)
        return accuracy, mean_similarity


Evaluation()
