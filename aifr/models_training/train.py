import os
import sys
import wandb
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from config import TRAIN
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)
from aifr.utils.model_handler import ModelHandler
from aifr.utils.trainer_handler import TrainerHandler
from aifr.utils.image_loader import ImageLoader


class TrainingLeaveOneOut:
    """
    Class for training a model using the leave-one-out cross-validation approach.
    """
    def __init__(self):
        """
        Initializes the TrainingLeaveOneOut class by loading the configuration and dataset,
        then starts the leave-one-out training process.
        """
        self.config = TRAIN.instance()
        self.dataset = self.load_dataset()
        self.train_leave_one_out()

    def load_dataset(self):
        """
        Loads the dataset based on the provided configuration.

        Returns:
            ImageLoader: The loaded dataset.
        """
        dataset_path = self.config.get('data_root')
        if not os.path.exists(dataset_path):
            print(f"Dataset path {dataset_path} does not exist.")
            return None
        return ImageLoader(root=dataset_path, transform=self.get_transforms())

    def get_transforms(self):
        """
        Defines and returns the transformations to be applied to the dataset images.

        Returns:
            transforms.Compose: The transformations to be applied.
        """
        return transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def train_leave_one_out(self):
        """
        Performs leave-one-out cross-validation training and evaluation.
        """
        if not self.dataset:
            print("Failed to load dataset.")
            return

        accuracy, accuracy_sum = 0, 0
        wandb.init(project=self.config['project'], entity=self.config['entity'])
        for leave_out_index in range(len(self.dataset)):
            print(f"Leave out index: {leave_out_index}")
            pretrained_model = ModelHandler().get_model(self.config['model_name'], self.config['dataset'], self.config['margin_loss'])
            model = ModelHandler().get_model(self.config['model_name'], self.config['eval_dataset'], self.config['margin_loss'])
            if self.config["model_weights"]:
                pretrained_model.load_state_dict(torch.load(self.config["model_weights"]), strict=False)
                print(f'Loaded weights from {self.config["model_weights"]}')
                pretrained_dict = {name: param for name, param in pretrained_model.named_parameters() if 'margin_loss.weights' not in name}
                model_dict = model.state_dict()
                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict)

            trainer = TrainerHandler().get_trainer(model, self.config)

            train_indices = [i for i in range(len(self.dataset)) if i != leave_out_index]
            test_indices = [leave_out_index]

            train_subset = Subset(self.dataset, train_indices)
            test_subset = Subset(self.dataset, test_indices)

            train_loader = DataLoader(train_subset, batch_size=self.config.get('batch_size'), shuffle=True)
            test_loader = DataLoader(test_subset, batch_size=1, shuffle=False)

            accuracy = trainer.train_leave_one_out(self.config['epochs'], train_loader, test_loader)
            accuracy_sum += accuracy
            print(f"Leave out index: {leave_out_index}, accuracy: {accuracy}")
            print(f"Average accuracy: {accuracy_sum/(leave_out_index+1)}")

        wandb.finish()


class Training:
    """
    Class for training a model using the specified configuration.
    """
    def __init__(self):
        """
        Initializes the Training class by loading the configuration and model,
        then starts the training process.
        """
        self.config = TRAIN.instance()

        wandb.init(project=self.config['project'], entity=self.config['entity'])

        wandb.config.update({
            'model_name': self.config['model_name'],
            'dataset': self.config['dataset'],
            'train_root': self.config['train_root'],
            'test_root': self.config['test_root'],
            'margin_loss': self.config['margin_loss'],
            'learning_rate': self.config['learning_rate'],
            'epochs': self.config['epochs'],
            'batch_size': self.config['batch_size'],
            'embedding_size': self.config['embedding_size'],
            'lambdas': self.config['lambdas'],
            'save_model': self.config['save_model'],
        })

        self.model = ModelHandler().get_model(self.config['model_name'], self.config['dataset'], self.config['margin_loss'])

        if self.config["model_weights"]:
            self.model.load_state_dict(torch.load(self.config["model_weights"]))
            print(f'Loaded weights from {self.config["model_weights"]}')

        self.train()

    def train(self):
        """
        Starts the training process using the loaded model and configuration.
        """
        self.trainer = TrainerHandler().get_trainer(self.model, self.config)

        wandb.watch(self.model, log="all")

        self.trainer.train(epochs=self.config["epochs"])

        artifact = wandb.Artifact("model", type="model")
        artifact.add_file(self.config["save_model"])
        wandb.log_artifact(artifact)
        wandb.finish()


#Training()
#TrainingLeaveOneOut()