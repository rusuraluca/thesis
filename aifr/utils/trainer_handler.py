import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)
from aifr.models.singletask.training import Singletask_Trainer
from aifr.models.multitask.training import Multitask_Trainer
from aifr.models.multitask_dal.training import Multitask_DAL_Trainer


class TrainerHandler:
    @staticmethod
    def get_trainer(model, config):
        """
        Retrieves the trainer instance based on the provided model and configuration.

        Args:
            model (nn.Module): The model instance.
            config (dict): The configuration dictionary containing 'model_name' among other settings.

        Returns:
            nn.Module: An instance of the specified trainer.

        Raises:
            ValueError: If the specified trainer is unsupported.
        """
        trainer = {
            'singletask': Singletask_Trainer,
            'multitask': Multitask_Trainer,
            'multitask_dal': Multitask_DAL_Trainer
        }

        if config['model_name'].lower() in trainer:
            return trainer[config['model_name'].lower()](
                model=model,
                config=config,
            )
        else:
            raise ValueError("Unsupported trainer.")
