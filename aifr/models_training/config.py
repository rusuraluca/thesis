class TRAIN(object):
    """
    TRAIN is a Singleton class that defines one instance of a dictionary with the configuration that the training process should use.
    """
    _instance = None  # class variable to hold the single instance of the class

    def __init__(self):
        """
        Initializes the TRAIN instance.

        This method raises a RuntimeError to prevent the instantiation of the class using the constructor.
        Instead, the class should be instantiated using the instance() class method.
        """
        raise RuntimeError('Call instance() instead')

    @classmethod
    def instance(cls):
        """
        Returns the single instance of the TRAIN class.

        This method checks if the single instance of the class (_instance) is None.
        If it is, it creates a new instance of the class and initializes it with a dictionary of configuration settings.
        If an instance already exists, it simply returns the existing instance.

        Returns:
            dict: A dictionary containing configuration settings for training.
        """
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            cls._instance = {
                'project': 'mpaifr',
                'entity': 'mpaifr',
                'model_name': 'multitask_dal', #singletask, multitask, multitask_dal
                'model_weights': 'aifr/models/multitask_dal/results/80-20small/best-model-93-89.pth',
                'dataset': 'small', #big, small
                'eval_dataset': 'fgnet',
                'data_root': './data/fgnet/',
                'train_root': './data/train_big',
                'test_root': './data/test_big',
                'margin_loss': 'cosface', #cosface, arcface
                'lambdas': (0.9, 0.9, 0.9),
                'batch_size': 64,
                'learning_rate': 0.01,
                'epochs': 40,
                'embedding_size': 512,
                'save_model': 'aifr/models/multitask_dal/results/loo',
            }
        return cls._instance
