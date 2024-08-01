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

    // Initialize Adam parameters
    InitializeAdam();
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

    // Calculate the loss for the current batch
    loss = CalculateLoss(targets, activations.back());
}

void NeuralNetwork::UpdateWeights(const std::string& optimizer) {
    if (optimizer == "Adam") {
        UpdateWeightsAdam();
    } else {
        UpdateWeightsSGD();
    }
}

void NeuralNetwork::UpdateWeightsSGD() {
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

void NeuralNetwork::UpdateWeightsAdam() {
    ++t;
    for (size_t i = 0; i < weights.size(); ++i) {
        for (int j = 0; j < layerSizes[i + 1]; ++j) {
            for (int k = 0; k < layerSizes[i]; ++k) {
                m_weights[i][j][k] = beta1 * m_weights[i][j][k] + (1.0 - beta1) * weightGradients[i][j][k];
                v_weights[i][j][k] = beta2 * v_weights[i][j][k] + (1.0 - beta2) * weightGradients[i][j][k] * weightGradients[i][j][k];
                double m_hat = m_weights[i][j][k] / (1.0 - std::pow(beta1, t));
                double v_hat = v_weights[i][j][k] / (1.0 - std::pow(beta2, t));
                weights[i][j][k] -= learningRate * m_hat / (std::sqrt(v_hat) + epsilon);
                weightGradients[i][j][k] = 0.0;
            }
            m_biases[i][j] = beta1 * m_biases[i][j] + (1.0 - beta1) * biasGradients[i][j];
            v_biases[i][j] = beta2 * v_biases[i][j] + (1.0 - beta2) * biasGradients[i][j] * biasGradients[i][j];
            double m_hat = m_biases[i][j] / (1.0 - std::pow(beta1, t));
            double v_hat = v_biases[i][j] / (1.0 - std::pow(beta2, t));
            biases[i][j] -= learningRate * m_hat / (std::sqrt(v_hat) + epsilon);
            biasGradients[i][j] = 0.0;
        }
    }
}


void NeuralNetwork::InitializeAdam() {
    m_weights.resize(weights.size());
    v_weights.resize(weights.size());
    m_biases.resize(biases.size());
    v_biases.resize(biases.size());

    for (size_t i = 0; i < weights.size(); ++i) {
        m_weights[i].resize(weights[i].size());
        v_weights[i].resize(weights[i].size());
        m_biases[i].resize(biases[i].size());
        v_biases[i].resize(biases[i].size());
        for (size_t j = 0; j < weights[i].size(); ++j) {
            m_weights[i][j].resize(weights[i][j].size(), 0.0);
            v_weights[i][j].resize(weights[i][j].size(), 0.0);
            m_biases[i][j] = 0.0;
            v_biases[i][j] = 0.0;
        }
    }
}


double NeuralNetwork::CalculateLoss(const std::vector<double>& targets, const std::vector<double>& outputs) {
    double sum = 0.0;
    for (size_t i = 0; i < targets.size(); ++i) {
        double diff = targets[i] - outputs[i];
        sum += diff * diff;
    }
    return sum / targets.size();
}

double NeuralNetwork::GetLoss() const {
    return loss;
}