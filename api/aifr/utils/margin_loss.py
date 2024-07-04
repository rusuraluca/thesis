import torch
import torch.nn as nn
import torch.nn.functional as functional


class MarginLoss:
    """
    A factory class for margin-based loss functions.

    This class provides a static method to retrieve a margin-based loss function
    based on the given name. Currently, it supports CosFace margin loss.

    Methods:
        get_margin_loss(margin_loss_name, number_of_classes, embedding_size):
            Returns an instance of the specified margin-based loss class.
    """
    @staticmethod
    def get_margin_loss(margin_loss_name, number_of_classes, embedding_size):
        """
        Retrieve a margin-based loss function based on the given name.

        Args:
            margin_loss_name (str): The name of the margin loss function.
            number_of_classes (int): The number of classes in the classification task.
            embedding_size (int): The size of the embeddings.

        Returns:
            nn.Module: An instance of the specified margin-based loss class.

        Raises:
            ValueError: If the specified margin loss name is unsupported.
        """
        margin_loss = {
            'cosface': CosFaceMarginLoss,
        }
        if margin_loss_name.lower() in margin_loss:
            return margin_loss[margin_loss_name.lower()](number_of_classes, embedding_size)
        else:
            raise ValueError("Unsupported loss head.")


class CosFaceMarginLoss(nn.Module):
    """
    CosFace Margin Loss introduced in:
    Z. Zhou X. Ji D. Gong J. Zhou Z. Li W. Liu H. Wang, Y. Wang.
    Cosface: Large margin cosine loss for deep face recognition. Conference on Computer
    Vision and Pattern Recognition (CVPR), 2018.

    This class implements the CosFace margin loss, which adds an angular margin to the logits
    to enhance discriminative power in face recognition tasks.

    Attributes:
        scale (float): The scale factor applied to the logits.
        margin (float): The margin value added to the logits.
        weights (nn.Parameter): The learnable weight parameters for the classification layer.
    """
    def __init__(self, number_of_classes, embedding_size=512, scale=32, margin=0.1):
        """
        Initialize the CosFaceMarginLoss.

        Args:
            number_of_classes (int): The number of classes in the classification task.
            embedding_size (int, optional): The size of the embeddings. Default is 512.
            scale (float, optional): The scale factor applied to the logits. Default is 32.
            margin (float, optional): The margin value added to the logits. Default is 0.1.
        """
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.weights = nn.Parameter(torch.Tensor(number_of_classes, embedding_size))
        nn.init.xavier_uniform_(self.weights) # initialize the weights with Xavier uniform distribution

    def forward(self, embeddings, labels):
        """
        Forward pass for the CosFace margin loss.

        Args:
            embeddings (torch.Tensor): The input embeddings of shape (batch_size, embedding_size).
            labels (torch.Tensor): The ground truth labels of shape (batch_size).

        Returns:
            torch.Tensor: The logits after applying the CosFace margin loss.
        """
        normalized_embeddings = functional.normalize(embeddings)
        normalized_weights = functional.normalize(self.weights)

        # compute the logits by taking the linear combination of embeddings and weights
        logits = functional.linear(normalized_embeddings, normalized_weights)

        if not self.training:
            return logits

        # create a margin tensor and scatter the margin values to the correct positions
        margins = torch.zeros_like(logits).scatter_(1, labels.unsqueeze(1), self.margin)
        logits -= margins  # subtract the margin from the logits

        logits *= self.scale # scale the logits

        return logits
