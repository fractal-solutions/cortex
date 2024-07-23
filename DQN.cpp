#include "DQN.h"
#include <algorithm>
#include <random>

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
