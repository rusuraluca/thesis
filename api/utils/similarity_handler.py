from PIL import Image
import torch
from torchvision import transforms as transforms
import torch.nn.functional as functional

class SimilarityHandler:
    @staticmethod
    def get_response(model, file1, file2):
        """
        Calculates the cosine similarity between two images using a given model.
        This method belongs to the class but does not require access to any instance-specific data.

        Args:
            model (torch.nn.Module): The deep learning model used to generate image embeddings.
            file1 (str): The path to the first image file.
            file2 (str): The path to the second image file.

        Returns:
            float: The cosine similarity score between the two images.

        Example:
            similarity = SimilarityHandler.get_response(model, 'path/to/image1.jpg', 'path/to/image2.jpg')
        """
        def load_image_from_file(file, transformation):
            """
            Loads and transforms an image from the given file path.

            Args:
                file (str): The path to the image file.
                transformation (torchvision.transforms.Compose): The transformations to apply to the image.

            Returns:
                torch.Tensor: The transformed image tensor ready for model input.
            """
            image = Image.open(file).convert('RGB')
            if transformation is not None:
                image = transformation(image)
                # add an extra dimension of size 1 at the 0th position to the image tensor as a batch dimension, resulting in a shape of [1, CHANNELS=3, HEIGHT, WIDTH]
                image = image.unsqueeze(0)
            return image

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

        image1 = load_image_from_file(file1, transform)
        image2 = load_image_from_file(file2, transform)
        image1_flipped = load_image_from_file(file1, transform_flipped)
        image2_flipped = load_image_from_file(file2, transform_flipped)

        with torch.no_grad():
            features1 = model(image1, return_embeddings=True)
            features2 = model(image2, return_embeddings=True)
            features1_flipped = model(image1_flipped, return_embeddings=True)
            features2_flipped = model(image2_flipped, return_embeddings=True)

        similarity = functional.cosine_similarity(features1 + features1_flipped, features2 + features2_flipped)
        return similarity.item()
