from models.model_config import ModelTypes


optimal_lrs = {
    "nd_exp": {
        "SGD": {20: 0.03},
        "Adam": {20: 0.01}
    },
    "nd": {
        "SGD": {20: 0.03},
        "Adam": {20: 0.01}
    },
    ModelTypes.NON_UNIFORM_SUM: {
        "SGD": {20: 0.0003},
        "Adam": {20: 0.01}
    },
    ModelTypes.ROTATION: {
        "SGD": {20: 0.003},
        "Adam": {20: 0.001}
    },
    ModelTypes.UNIFORM_SUM: {
        "SGD": {20: 0.001},
        "Adam": {20: 0.001}
    },
    ModelTypes.VAE: {
        "SGD": {20: 0.001},
        "Adam": {20: 0.001}
    }
}