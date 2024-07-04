import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)
from aifr.models.singletask.model import Singletask
from aifr.models.multitask.model import Multitask
from aifr.models.multitask_dal.model import Multitask_DAL


class ModelHandler:
    @staticmethod
    def get_model(model_name, dataset_name, margin_loss_name):
        """
        Retrieves the model instance based on the provided model name, dataset name, and margin loss name.
        This method belongs to the class but does not require access to any instance-specific data.

        Args:
            model_name (str): The name of the model ('singletask', 'multitask', 'multitask_dal').
            dataset_name (str): The name of the dataset ('big', 'small', 'fgnet').
            margin_loss_name (str): The name of the margin loss function.

        Returns:
            nn.Module: An instance of the specified model.

        Raises:
            ValueError: If the specified model name is unsupported.
        """
        model_type = {
            'singletask': Singletask,
            'multitask': Multitask,
            'multitask_dal': Multitask_DAL
        }

        dataset = {
            'big': 1035,
            'small': 500,
            'fgnet': 82,
        }

        if model_name.lower() in model_type:
            return model_type[model_name.lower()](
                embedding_size=512,
                number_of_classes=dataset[dataset_name],
                margin_loss_name=margin_loss_name,
            )
        else:
            raise ValueError("Unsupported model.")
