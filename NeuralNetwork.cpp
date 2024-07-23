#include "NeuralNetwork.h"
#include <cmath>
#include <random>

NeuralNetwork::NeuralNetwork(int inputSize, const std::vector<int>& hiddenLayers, int outputSize,
                             std::function<double(double)> activation, 
                             std::function<double(double)> activationDerivative) 
    : activation(activation), activationDerivative(activationDerivative) {
    layerSizes.push_back(inputSize);
    layerSizes.insert(layerSizes.end(), hiddenLayers.begin(), hiddenLayers.end());
    layerSizes.push_back(outputSize);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    for (size_t i = 1; i < layerSizes.size(); ++i) {
        weights.push_back(std::vector<std::vector<double>>(layerSizes[i], std::vector<double>(layerSizes[i - 1])));
        biases.push_back(std::vector<double>(layerSizes[i]));
        weightGradients.push_back(std::vector<std::vector<double>>(layerSizes[i], std::vector<double>(layerSizes[i - 1], 0.0)));
        biasGradients.push_back(std::vector<double>(layerSizes[i], 0.0));
        for (int j = 0; j < layerSizes[i]; ++j) {
            biases.back()[j] = dis(gen);
            for (int k = 0; k < layerSizes[i - 1]; ++k) {
                weights.back()[j][k] = dis(gen);
            }
        }
    }
}

std::vector<double> NeuralNetwork::Forward(const std::vector<double>& inputs) {
    std::vector<double> a = inputs;
    for (size_t i = 0; i < weights.size(); ++i) {
        std::vector<double> z(layerSizes[i + 1]);
        for (int j = 0; j < layerSizes[i + 1]; ++j) {
            z[j] = biases[i][j];
            for (int k = 0; k < layerSizes[i]; ++k) {
                z[j] += weights[i][j][k] * a[k];
            }
            a[j] = activation(z[j]);
        }
    }
    return a;
}

void NeuralNetwork::Backward(const std::vector<double>& inputs, const std::vector<double>& targets, double learningRate) {
    std::vector<std::vector<double>> activations(layerSizes.size());
    std::vector<std::vector<double>> zs(layerSizes.size());

    // Forward pass
    std::vector<double> a = inputs;
    activations[0] = inputs;
    for (size_t i = 0; i < weights.size(); ++i) {
        std::vector<double> z(layerSizes[i + 1]);
        for (int j = 0; j < layerSizes[i + 1]; ++j) {
            z[j] = biases[i][j];
            for (int k = 0; k < layerSizes[i]; ++k) {
                z[j] += weights[i][j][k] * a[k];
            }
            a[j] = activation(z[j]);
        }
        activations[i + 1] = a;
        zs[i] = z;
    }

    // Backward pass
    std::vector<double> delta(layerSizes.back());
    for (int i = 0; i < layerSizes.back(); ++i) {
        delta[i] = activations.back()[i] - targets[i];
    }

    for (int i = weights.size() - 1; i >= 0; --i) {
        std::vector<double> deltaNext(layerSizes[i]);
        for (int j = 0; j < layerSizes[i]; ++j) {
            deltaNext[j] = 0.0;
            for (int k = 0; k < layerSizes[i + 1]; ++k) {
                deltaNext[j] += delta[k] * weights[i][k][j];
            }
            deltaNext[j] *= activationDerivative(zs[i][j]);
        }
        for (int j = 0; j < layerSizes[i + 1]; ++j) {
            for (int k = 0; k < layerSizes[i]; ++k) {
                weightGradients[i][j][k] += delta[j] * activations[i][k];
            }
            biasGradients[i][j] += delta[j];
        }
        delta = deltaNext;
    }
}

void NeuralNetwork::UpdateWeights() {
    for (size_t i = 0; i < weights.size(); ++i) {
        for (int j = 0; j < layerSizes[i + 1]; ++j) {
            for (int k = 0; k < layerSizes[i]; ++k) {
                weights[i][j][k] -= weightGradients[i][j][k] * learningRate;
                weightGradients[i][j][k] = 0.0;
            }
            biases[i][j] -= biasGradients[i][j] * learningRate;
            biasGradients[i][j] = 0.0;
        }
    }
}
