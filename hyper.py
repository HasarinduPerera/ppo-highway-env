class Hyperparameters():

    # Define Device
    DEVICE = 'cpu'

    # Define training params
    N_EPISODES = 300
    PRINT_FREQ = 20

    # Define agent params
    POLICY_LR = 3E-4
    VALUE_LR = 1E-3
    TARGET_KL_DIV = 0.02
    MAX_POLICY_TRAIN_ITERS = 80
    VALUE_TRAIN_ITERS = 80

    # Define observation and action space
    OBS_SPACE = 847
    ACTION_SPACE = 5
    ACTION_TYPE = 'DiscreteMetaAction'
    OBS_TYPE = 'OccupancyGrid'
    

    @classmethod
    def all(cls):
        return [value for name, value in vars(cls).items() if name.isupper()]