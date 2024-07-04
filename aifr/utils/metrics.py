import torch

def compute_accuracy(predictions, labels):
    """
    Computes the accuracy of predictions against the ground truth labels.

    Args:
        predictions (torch.Tensor): The predicted labels.
        labels (torch.Tensor): The ground truth labels.

    Returns:
        torch.Tensor: The mean accuracy.
    """
    # disable gradient calculation for accuracy computation
    with torch.no_grad():
        # compare predictions with labels and compute the mean accuracy
        return (predictions.squeeze() == labels.squeeze()).float().mean()

def compute_loss(id_loss, age_loss=0, gender_loss=0, cano_corr=0, lambdas=(1, 1, 1)):
    """
    Computes the total loss by combining identity loss, age loss, gender loss, and canonical correlation.

    Args:
        id_loss (torch.Tensor): The identity loss.
        age_loss (torch.Tensor, optional): The age loss. Default is 0.
        gender_loss (torch.Tensor, optional): The gender loss. Default is 0.
        cano_corr (torch.Tensor, optional): The canonical correlation loss. Default is 0.
        lambdas (tuple, optional): The weights for the age loss, gender loss, and canonical correlation. Default is (1, 1, 1).

    Returns:
        torch.Tensor: The total loss.
    """
    return id_loss + lambdas[0] * age_loss + lambdas[1] * gender_loss + lambdas[2] * cano_corr # combine the losses using the provided weights (lambdas)
