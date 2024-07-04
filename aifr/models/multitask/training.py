import os
import sys
from itertools import chain
import torch
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.append(parent_dir)
from aifr.utils.metrics import compute_loss
from aifr.utils.image_loader import ImageLoader


class Multitask_Trainer:
    def __init__(self, model, config):
        """
        Initializes the Multitask_Trainer with the specified model and configuration.

        Args:
            model (nn.Module): The model to be trained.
            config (dict): Configuration dictionary containing training parameters.
        """
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

        self.model = model.to(self.device)
        self.config = config

        self.learning_rate = self.config['learning_rate']
        self.batch_size = self.config['batch_size']
        self.lambdas =  self.config['lambdas']

        self.optimizer = optim.SGD(
            params=chain(
                self.model.margin_loss.parameters(),
                self.model.age_classifier.parameters(),
                self.model.gender_classifier.parameters(),
                self.model.frfm.parameters(),
                self.model.backbone.parameters(),
            ),
            lr=self.learning_rate,
            momentum=0.9,
        )

        self.transforms_train = transforms.Compose([
            transforms.RandomResizedCrop((160, 160)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.transforms_test = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.CenterCrop(160),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.set_train_mode(True)

    @staticmethod
    def set_grads(mod, state):
        """
        Set the requires_grad attribute for all parameters in a module.

        Args:
            mod (nn.Module): The module whose parameters' requires_grad attribute is to be set.
            state (bool): The state to set for the requires_grad attribute.
        """
        for param in mod.parameters():
            param.requires_grad = state

    def set_train_mode(self, state):
        """
        Set the model's training mode and requires_grad attributes for its parameters.

        Args:
            state (bool): The state to set for the training mode and requires_grad attribute.
        """
        self.set_grads(self.model.margin_loss, state)
        self.set_grads(self.model.age_classifier, state)
        self.set_grads(self.model.gender_classifier, state)
        self.set_grads(self.model.frfm, state)
        self.set_grads(self.model.backbone, state)

    def load_data(self, training=True):
        """
        Load the dataset based on the training flag and apply the appropriate transformations.

        Args:
            training (bool, optional): If True, load the training dataset. Otherwise, load the testing dataset. Default is True.

        Returns:
            DataLoader: DataLoader object for the dataset.
        """
        path_key = 'train_root' if training else 'test_root'
        dataset_path = self.config.get(path_key)

        if dataset_path is None or not os.path.exists(dataset_path):
            print(f"Dataset path {dataset_path} does not exist.")
            return None

        if training:
            dataset = ImageLoader(
                root=dataset_path,
                transform=self.transforms_train)
        else:
            dataset = ImageLoader(
                root=dataset_path,
                transform=self.transforms_test)

        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def save_model(self, epoch):
        """
        Save the model's state dictionary to a file.

        Args:
            epoch (int): The current epoch number.
        """
        model_path = os.path.join(self.config['save_model'], 'model_epoch_{}.pth'.format(epoch))
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    def run_epoch(self, loader, training):
        """
        Run a single epoch of training or testing.

        Args:
            loader (DataLoader): DataLoader object for the dataset.
            training (bool): If True, run training. Otherwise, run testing.
        """
        if loader is None:
            print("Warning: DataLoader is None. Skipping this epoch.")
            return

        phase = 'Training' if training else 'Testing'
        self.model.train() if training else self.model.eval()

        for i, (images, labels, age_groups, genders) in enumerate(loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            age_groups = age_groups.to(self.device)
            genders = genders.to(self.device)


            if training:
                id_loss, id_accuracy, age_loss, age_accuracy, gender_loss, gender_accuracy \
                    = self.model(inputs=images, labels=labels, age_groups=age_groups, genders=genders)

                total_loss = compute_loss(
                    id_loss,
                    age_loss,
                    gender_loss,
                    lambdas=self.lambdas
                )

            else:
                with torch.no_grad():
                    id_loss, id_accuracy, age_loss, age_accuracy, gender_loss, gender_accuracy \
                        = self.model(inputs=images, labels=labels, age_groups=age_groups, genders=genders)

                    total_loss = compute_loss(
                        id_loss,
                        age_loss,
                        gender_loss,
                        lambdas=self.lambdas
                    )

            if training:
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

            metrics = {
                f"{phase}/total_loss": total_loss.item(),
                f"{phase}/id_loss": id_loss.item(),
                f"{phase}/id_accuracy": id_accuracy.item(),
                f"{phase}/age_loss": age_loss.item(),
                f"{phase}/age_accuracy": age_accuracy.item(),
                f"{phase}/gender_loss": gender_loss.item(),
                f"{phase}/gender_accuracy": gender_accuracy.item(),
                f"{phase}/progress": i / len(loader)
            }

            wandb.log(metrics)

    def train(self, epochs):
        """
        Train the model for a specified number of epochs.

        Args:
            epochs (int): Number of epochs to train the model.
        """
        train_loader = self.load_data(training=True)
        test_loader = self.load_data(training=False)

        for epoch in range(epochs):
            print("Training")
            self.run_epoch(train_loader, training=True)

            if test_loader:
                print("Testing")
                self.run_epoch(test_loader, training=False)

            self.save_model(epoch)