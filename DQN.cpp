#include "DQN.h"
#include <algorithm>
#include <random>
#include <fstream>
#include <iostream>

// Constructor
DQN::DQN(int stateSize, int actionSize, const std::vector<int>& hiddenLayers,
         std::function<double(double)> activation, 
         std::function<double(double)> activationDerivative)
    : stateSize(stateSize), actionSize(actionSize), epsilon(1.0), gamma(0.99),
      epsilonDecay(0.995), epsilonMin(0.1), learningRate(0.001),
      qNetwork(stateSize, hiddenLayers, actionSize, activation, activationDerivative), 
      targetNetwork(stateSize, hiddenLayers, actionSize, activation, activationDerivative),
      memoryCapacity(1000), memoryIndex(0), rng(std::random_device{}()), indexDistribution(0, memoryCapacity - 1) {
    memory.resize(memoryCapacity);
    UpdateTargetNetwork();
}

// Select action based on epsilon-greedy strategy
int DQN::SelectAction(const std::vector<double>& state, double epsilon) {
    if (static_cast<double>(rand()) / RAND_MAX < epsilon) {
        return rand() % actionSize;
    } else {
        std::vector<double> qValues = GetQValues(state);
        return std::distance(qValues.begin(), std::max_element(qValues.begin(), qValues.end()));
    }
}

// Train the network using the stored experiences
void DQN::Train(const std::vector<double>& state, int action, double reward, const std::vector<double>& nextState, double gamma, double epsilonDecay) {
    memory[memoryIndex] = std::make_tuple(state, action, reward, nextState);
    memoryIndex = (memoryIndex + 1) % memoryCapacity;

    if (epsilon > epsilonMin) {
        epsilon *= epsilonDecay;
    }

    if (memoryIndex >= memoryCapacity / 10) {
        for (size_t i = 0; i < memoryCapacity / 10; ++i) {
            size_t idx = indexDistribution(rng); // Generate random index
            auto& experience = memory[idx];
            std::vector<double> targetQ = GetQValues(std::get<0>(experience));
            if (!std::get<3>(experience).empty()) {
                std::vector<double> nextQ = GetQValues(std::get<3>(experience));
                targetQ[std::get<1>(experience)] = std::get<2>(experience) + gamma * (*std::max_element(nextQ.begin(), nextQ.end()));
            } else {
                targetQ[std::get<1>(experience)] = std::get<2>(experience);
            }
            qNetwork.Backward(std::get<0>(experience), targetQ, learningRate);
        }
        //THIS DQN IMPLEMENTATION HAS BEEEN FORCED TO USE ADAM, SHOULD OPTIMIZE CODE SUCH THAT THERE IS OPTION OF ADAM OR SGD
        // Update weights after processing the mini-batch
        qNetwork.UpdateWeights("Adam");
    }
}


// Update the target network with the Q-network weights
void DQN::UpdateTargetNetwork() {
    targetNetwork = qNetwork;
}

// Get Q-values for a given state
std::vector<double> DQN::GetQValues(const std::vector<double>& state) {
    return qNetwork.Forward(state);
}


int DQN::SaveModel(const std::string& filepath) {
    std::ofstream file(filepath, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Unable to open file for saving model." << std::endl;
        return 1;
    }

    file.write(reinterpret_cast<const char*>(stateSize), sizeof(stateSize));
    file.write(reinterpret_cast<const char*>(actionSize), sizeof(actionSize));
    // Serialize weights and biases
    for (const auto& layer : qNetwork.weights) {
        for (const auto& neuron : layer) {
            file.write(reinterpret_cast<const char*>(neuron.data()), neuron.size() * sizeof(double));
        }
    }
    for (const auto& layer : qNetwork.biases) {
        file.write(reinterpret_cast<const char*>(layer.data()), layer.size() * sizeof(double));
    }
    file.close();
    return 0;
}

int DQN::LoadModel(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Unable to open file for loading model." << std::endl;
        return 1;
    }

    file.read(reinterpret_cast<char*>(stateSize), sizeof(stateSize));
    file.read(reinterpret_cast<char*>(actionSize), sizeof(actionSize));
    // Deserialize weights and biases
    for (auto& layer : qNetwork.weights) {
        for (auto& neuron : layer) {
            file.read(reinterpret_cast<char*>(neuron.data()), neuron.size() * sizeof(double));
        }
    }
    for (auto& layer : qNetwork.biases) {
        file.read(reinterpret_cast<char*>(layer.data()), layer.size() * sizeof(double));
    }
    file.close();
    return 0;
}