class ModelTypes:
    NON_UNIFORM_SUM = "non_uniform_sum"
    UNIFORM_SUM = "uniform_sum"
    ROTATION = "rotation"
    NESTED_DROPOUT = "nested_dropout"
    VAE = "vae"
    VALID_MODEL_TYPES = (NON_UNIFORM_SUM, UNIFORM_SUM, ROTATION, NESTED_DROPOUT, VAE)


class ModelConfig:
    def __init__(self, model_name, model_type, model_class, input_dim, hidden_dim, init_scale, optim_class, lr,
                 extra_model_args={}, extra_optim_args={}):
        self.model_name = model_name
        self.model_type = model_type
        assert model_type in ModelTypes.VALID_MODEL_TYPES
        self.model_class = model_class
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.init_scale = init_scale
        self.extra_model_args = extra_model_args

        self.optim_class = optim_class
        self.lr = lr
        self.extra_optim_args = extra_optim_args

        self.model = model_class(input_dim=input_dim, hidden_dim=hidden_dim, init_scale=init_scale, **extra_model_args).cuda()

        self.optimizer = optim_class(self.model.parameters(), lr=lr, **extra_optim_args)

    @property
    def name(self):
        return self.model_name

    @property
    def type(self):
        return self.model_type

    def get_model(self):
        return self.model

    def get_optimizer(self):
        return self.optimizer
