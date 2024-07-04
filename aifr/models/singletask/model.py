import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as functional
from facenet_pytorch import InceptionResnetV1

from ...utils.margin_loss import MarginLoss
from ...utils.metrics import compute_accuracy


class Singletask(nn.Module):
    """
    Singletask model,
    integrates the Backbone CNN or the InceptionResnetV1 CNN for feature extraction,
    utilizes specific margin-based loss functions (e.g., CosFace, ArcFace) for enhancing discriminative power.
    """
    def __init__(self, embedding_size=512, number_of_classes=500, margin_loss_name='cosface', initializer=None):
        """
        Initializes the Singletask model with the specified parameters.

        Args:
            embedding_size (int, optional): Size of the embedding vector. Default is 512.
            number_of_classes (int, optional): Number of classes for classification. Default is 500.
            margin_loss_name (str, optional): Name of the margin loss function to use. Default is 'cosface'.
            initializer (callable, optional): Initializer for model weights. Default is None.
        """
        super(Singletask, self).__init__()

        self.backbone = InceptionResnetV1(pretrained='vggface2', dropout_prob=0.4)
        self.margin_loss = MarginLoss().get_margin_loss(margin_loss_name, number_of_classes, embedding_size)
        self.id_criterion = nn.CrossEntropyLoss()

    def forward(self, inputs, labels=None, return_embeddings=False):
        """
        Forward pass for the Singletask model.

        Args:
            inputs (torch.Tensor): Input tensor containing the images.
            labels (torch.Tensor, optional): Ground truth labels for the images. Default is None.
            return_embeddings (bool, optional): If True, returns the normalized embeddings instead of the loss and accuracy. Default is False.

        Returns:
            torch.Tensor: If return_embeddings is True, returns the normalized embeddings.
            tuple: If return_embeddings is False, returns a tuple containing the identity loss and accuracy.
        """
        embeddings = self.backbone(inputs)

        if return_embeddings:
            return functional.normalize(embeddings)

        id_logits = self.margin_loss(embeddings, labels)
        id_loss = self.id_criterion(id_logits, labels)
        id_accuracy = compute_accuracy(torch.max(id_logits, dim=1)[1], labels)

        return id_loss, id_accuracy
