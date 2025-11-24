import numpy as np

class RandomFeatureGenerator_zx:
    # number of parameters expected for each distribution
    distribution_requirements = {
        "standard_normal": 0,
        # add more if you need them later
    }

    permitted_activation_functions = [
        "cos",
        "sin",
        "exp",
        "arctan",
        "tanh",
    ]

    def __init__(self):
        pass

    @staticmethod
    def check_distribution_requirements(distribution: str, distribution_parameters: list):
        if distribution not in RandomFeatureGenerator_zx.distribution_requirements:
            raise Exception(
                f"{distribution} is not permitted. If you need it, update distribution_requirements."
            )
        required = RandomFeatureGenerator_zx.distribution_requirements[distribution]
        if len(distribution_parameters) != required:
            raise Exception(
                f"{distribution} requires {required} parameters, "
                f"but got {len(distribution_parameters)}"
            )

    @staticmethod
    def generate_random_neuron_features(
        features: np.ndarray,
        random_seed: int,
        distribution: str,
        distribution_parameters: list,
        activation: str,
        number_features: int,
    ) -> np.ndarray:
        """
        Builds random neuron features f(W'x) where W has random rows, and f is an activation.
        features: (n, d)
        returns: (n, number_features)
        """
        np.random.seed(random_seed)
        signals = features

        # check distribution is allowed
        RandomFeatureGenerator_zx.check_distribution_requirements(
            distribution, distribution_parameters
        )

        n, d = signals.shape
        size = [d + 1, number_features]  # +1 for bias term

        # draw random weight vectors
        random_vectors = getattr(np.random, distribution)(
            *distribution_parameters, size
        )  # e.g. standard_normal → np.random.standard_normal(size)

        # normalize each column (each weight vector) to have norm 1
        norms = np.linalg.norm(random_vectors, axis=0, keepdims=True)
        norms[norms == 0] = 1e-12
        random_vectors = random_vectors / norms
        
        bias = random_vectors[-1, :]  # bias terms
        coefficients = random_vectors[:-1, :]  # weight vectors

        # W'x + b → (number_features, n)
        transformed_signals = np.matmul(coefficients.T, signals.T) + bias[:, None]
        # apply activation
        if activation in ["cos", "sin", "exp", "arctan", "tanh"]:
            final_random_features = getattr(np, activation)(transformed_signals)
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # return (n, number_features)
        return final_random_features.T