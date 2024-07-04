import torch
import torch.nn as nn
import torch.nn.functional as functional

from ...utils.margin_loss import MarginLoss
from facenet_pytorch import InceptionResnetV1


class FRFM(nn.Module):
    """
    Feature Residual Factorization Module (FRFM) to map the initial face features
    to form the age-dependent feature through FC + ReLU
    and the gender-dependent feature through FC +ReLU,
    and the residual part is regarded as the identity-dependent feature.
    """
    def __init__(self, embedding_size=512):
        super(FRFM, self).__init__()
        self.age_transform = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            nn.ReLU(inplace=True),
        )

        self.gender_transform = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            nn.ReLU(inplace=True),
        )

    def forward(self, embeddings):
        """
        Forward pass for the FRFM.

        Args:
            embeddings (torch.Tensor): Input embeddings.

        Returns:
            tuple: Identity, age, and gender embeddings.
        """
        age_embeddings = self.age_transform(embeddings)
        gender_embeddings = self.gender_transform(embeddings)
        id_embeddings = embeddings - age_embeddings - gender_embeddings
        return id_embeddings, age_embeddings, gender_embeddings


class BCCM(nn.Module):
    """
    Batch Canonical Correlation Mapping Module (BCCM) introduced in
    Hao Wang, Dihong Gong, Zhifeng Li, and Wei Liu.
    Decorrelated adversarial learning for age-invariant face recognition, 2019
    modified with the gender-reliant features.
    """
    def __init__(self, embedding_size=512):
        """
        Initializes the BCCA with linear layers for projecting.

        Args:
            embedding_size (int, optional): Size of the embedding vector. Default is 512.
        """
        super(BCCM, self).__init__()
        self.id_predictor = nn.Linear(embedding_size, 1, bias=False)
        self.age_predictor = nn.Linear(embedding_size, 1, bias=False)
        self.gender_predictor = nn.Linear(embedding_size, 1, bias=False)

    def forward(self, id_features, age_features, gender_features):
        """
        Forward pass for the BCCA.

        Args:
            id_features (torch.Tensor): Identity features.
            age_features (torch.Tensor): Age features.
            gender_features (torch.Tensor): Gender features.

        Returns:
            torch.Tensor: Correlation coefficient between identity, age, and gender embeddings.
        """
        id_predictions = self.id_predictor(id_features)
        age_predictions = self.age_predictor(age_features)
        gender_predictions = self.gender_predictor(gender_features)

        id_mean = id_predictions.mean(dim=0)
        age_mean = age_predictions.mean(dim=0)
        gender_mean = gender_predictions.mean(dim=0)

        id_var = id_predictions.var(dim=0) + 1e-6
        age_var = age_predictions.var(dim=0) + 1e-6
        gender_var = gender_predictions.var(dim=0) + 1e-6

        id_age_corr = ((age_predictions - age_mean) * (id_predictions - id_mean)).mean(dim=0).pow(2) / (
                age_var * id_var)
        id_gender_corr = ((gender_predictions - gender_mean) * (id_predictions - id_mean)).mean(dim=0).pow(2) / (
                gender_var * id_var)
        age_gender_corr = ((age_predictions - age_mean) * (gender_predictions - gender_mean)).mean(dim=0).pow(2) / (
                age_var * gender_var)

        correlation_coefficient = (id_age_corr + id_gender_corr + age_gender_corr)/3

        return correlation_coefficient


class Multitask_DAL(nn.Module):
    """
    Multitask and Decorrelated Adversarial Learning (DAL) model,
    integrates the Backbone CNN or the InceptionResnetV1 CNN for feature extraction,
    integrates Feature Residual Factorization Module (FRFM) to separate age, gender, and identity features,
    utilizes specific margin-based loss functions (e.g., CosFace, ArcFace) for enhancing discriminative power,
    employs the BCCM to minimize correlation between age, gender and identity features.
    Similar to the one introduced in
    Hao Wang, Dihong Gong, Zhifeng Li, and Wei Liu.
    Decorrelated adversarial learning for age-invariant face recognition, 2019.
    """
    def __init__(self, embedding_size=512, number_of_classes=1035, margin_loss_name='cosface', initializer=None):
        super(Multitask_DAL, self).__init__()

        self.backbone = InceptionResnetV1(pretrained='vggface2', dropout_prob=0.4)
        self.margin_loss = MarginLoss().get_margin_loss(margin_loss_name, number_of_classes, embedding_size)
        self.frfm = FRFM(embedding_size)
        self.bcca = BCCM(embedding_size)

        self.age_classifier = nn.Sequential(
            nn.Linear(embedding_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            nn.Linear(256, 3),
        )

        self.gender_classifier = nn.Sequential(
            nn.Linear(embedding_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            nn.Linear(256, 2),
        )

        self.id_criterion = nn.CrossEntropyLoss()
        self.age_criterion = nn.CrossEntropyLoss()
        self.gender_criterion = nn.CrossEntropyLoss()

    def forward(self, inputs, labels=None, age_groups=None, genders=None, return_embeddings=False):
        embeddings = self.backbone(inputs)
        id_embeddings, _, _ = self.frfm(embeddings)

        if return_embeddings:
            return functional.normalize(id_embeddings)
