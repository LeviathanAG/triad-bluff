"""Tiny residual neural Bluff scorer with standard-library inference."""
import math
from learned_policy_v5 import Policy as Features, FEATURE_NAMES


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

