"""Tiny residual neural Bluff scorer with standard-library inference."""
import math
from bluff_policy_v5 import Policy as Features, FEATURE_NAMES


class Policy(Features):
    def __init__(self, document):
        if document.get('format') != 'triad-bluff-residual-mlp-v1' or document.get('features') != FEATURE_NAMES:
            raise ValueError('incompatible neural Bluff schema')
        base = document['base_weights']
        super().__init__(base)
        self.hidden = document['hidden_weight']; self.bias = document['hidden_bias']
        self.output = document['output_weight']; self.output_bias = float(document['output_bias'])
        self.means = document['feature_mean']; self.scales = document['feature_scale']
        width = len(self.hidden)
        if not 1 <= width <= 128 or len(self.bias) != width or len(self.output) != width:
            raise ValueError('invalid hidden layer shape')
        if any(len(row) != len(FEATURE_NAMES) for row in self.hidden) or len(self.means) != len(FEATURE_NAMES) or len(self.scales) != len(FEATURE_NAMES):
            raise ValueError('invalid neural input shape')
        parameters = base + self.bias + self.output + self.means + self.scales + [self.output_bias]
        parameters += [weight for row in self.hidden for weight in row]
        if not all(math.isfinite(float(weight)) for weight in parameters) or any(scale <= 0. for scale in self.scales):
            raise ValueError('nonfinite neural parameters or invalid feature scale')

    def score_features(self, features):
        normalized = [(value - mean) / scale for value, mean, scale in zip(features, self.means, self.scales)]
        residual = self.output_bias
        for row, bias, output in zip(self.hidden, self.bias, self.output):
            hidden = max(0., bias + sum(weight * value for weight, value in zip(row, normalized)))
            residual += output * hidden
        return sum(weight * value for weight, value in zip(self.weights, features)) + residual

    def act(self, observation):
        self.update(observation)
        return max(self.candidates(observation), key=lambda item: self.score_features(item[1]))[0]


class FastPolicy(Policy):
    """Training-only NumPy batching; near ties use exact deployment scoring."""
    def __init__(self, document):
        super().__init__(document)
        import numpy as np
        self.np = np
        self.matrix = np.asarray(self.hidden, dtype=np.float64)
        self.hidden_bias = np.asarray(self.bias, dtype=np.float64)
        self.output_array = np.asarray(self.output, dtype=np.float64)
        self.base_array = np.asarray(self.weights, dtype=np.float64)
        self.mean_array = np.asarray(self.means, dtype=np.float64)
        self.scale_array = np.asarray(self.scales, dtype=np.float64)

    def act(self, observation):
        self.update(observation)
        candidates = self.candidates(observation)
        matrix = self.np.asarray([vector for _, vector in candidates], dtype=self.np.float64)
        scores = matrix @ self.base_array + self.output_bias
        scores += self.np.maximum(0., ((matrix - self.mean_array) / self.scale_array) @ self.matrix.T + self.hidden_bias) @ self.output_array
        largest = float(scores.max())
        band = 1e-8 * max(1., abs(largest))
        finalists = [index for index, value in enumerate(scores) if value >= largest - band]
        chosen = max(finalists, key=lambda index: self.score_features(candidates[index][1]))
        return candidates[chosen][0]
