from models.model_config import ModelTypes

hdims = [2, 5, 10, 20, 50, 100, 200, 300, 400, 500]

optimal_lrs = {
    "nd_exp": {
        "SGD": {dim: 0.001 for dim in hdims},
        "Adam": {
            2: 0.003,
            5: 0.003,
            10: 0.003,
            20: 0.003,
            50: 0.003,
            100: 0.01,
            200: 0.01,
            300: 0.01,
            400: 0.01,
            500: 0.01
        }
    },
    "nd": {
        "SGD": {dim: 0.001 for dim in hdims},
        "Adam": {
            2: 0.01,
            5: 0.01,
            10: 0.01,
            20: 0.01,
            50: 0.003,
            100: 0.003,
            200: 0.003,
            300: 0.003,
            400: 0.003,
            500: 0.003
        }
    },
    ModelTypes.NON_UNIFORM_SUM: {
        "SGD": {dim: 0.001 for dim in hdims},
        "Adam": {dim: 0.003 for dim in hdims}
    },
    ModelTypes.ROTATION: {
        "SGD": {
            2: 0.0001,
            5: 0.0001,
            10: 0.0001,
            20: 0.0001,
            50: 0.0001,
            100: 0.0001,
            200: 0.0001,
            300: 0.0001,
            400: 0.0001,
            500: 0.0001
        },
        "Adam": {dim: 0.0003 for dim in hdims}
    },
    ModelTypes.UNIFORM_SUM: {
        "SGD": {dim: 0.001 for dim in hdims},
        "Adam": {dim: 0.003 for dim in hdims}
    },
    ModelTypes.VAE: {
        "SGD": {dim: 0.0003 for dim in hdims},
        "Adam": {
            2: 0.003,
            5: 0.003,
            10: 0.003,
            20: 0.003,
            50: 0.001,
            100: 0.001,
            200: 0.001,
            300: 0.001,
            400: 0.001,
            500: 0.001
        }
    }
}