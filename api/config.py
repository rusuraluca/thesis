class CONFIG(object):
    """
    CONFIG is a Singleton class that defines one instance of a dictionary with the configuration that the API should use.
    """
    _instance = None

    def __init__(self):
        """
        Initializes the CONFIG instance.

        This method raises a RuntimeError to prevent the instantiation of the class using the constructor.
        Instead, the class should be instantiated using the instance() class method.
        """
        raise RuntimeError('Call instance() instead')

    @classmethod
    def instance(cls):
        """
        Returns the single instance of the CONFIG class.

        This method checks if the single instance of the class (_instance) is None.
        If it is, it creates a new instance of the class and initializes it with a dictionary of configuration settings.
        If an instance already exists, it simply returns the existing instance.

        Returns:
            dict: A dictionary containing configuration settings.
        """
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            cls._instance = {
                'model_name': 'multitask_dal', #singletask, multitask, multitask_dal
                'dataset': 'small', #big, small
                'model_weights': 'aifr/models/multitask_dal/results/80-20small/best-model-93-89.pth',
                'margin_loss': 'cosface', #cosface
            }
        return cls._instance
