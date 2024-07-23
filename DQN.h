#ifndef DQN_H
#define DQN_H

#include <vector>
#include <tuple>
#include <functional>
#include <random>
#include "NeuralNetwork.h"

class DQN {
public:
    DQN(int stateSize, int actionSize, const std::vector<int>& hiddenLayers,
        std::function<double(double)> activation = [](double x) { return x > 0 ? x : 0; }, // Default ReLU
        std::function<double(double)> activationDerivative = [](double x) { return x > 0 ? 1 : 0; }); // Default ReLU derivative

    int SelectAction(const std::vector<double>& state, double epsilon);
    void Train(const std::vector<double>& state, int action, double reward, const std::vector<double>& nextState, double gamma, double epsilonDecay);
    void UpdateTargetNetwork();
    std::vector<double> GetQValues(const std::vector<double>& state);

    // Public getters for epsilon and gamma
    double GetEpsilon() const { return epsilon; }
    double GetGamma() const { return gamma; }
    double GetEpsilonDecay() const { return epsilonDecay; }

    // Save and load model methods
    int SaveModel(const std::string& filepath);
    int LoadModel(const std::string& filepath);

private:
    int stateSize;
    int actionSize;
    double epsilon;
    double gamma;
    double epsilonDecay;
    double epsilonMin;
    double learningRate;
    NeuralNetwork qNetwork;
    NeuralNetwork targetNetwork;
    size_t memoryCapacity;
    size_t memoryIndex;
    std::vector<std::tuple<std::vector<double>, int, double, std::vector<double>>> memory;

    std::mt19937 rng; // Random number generator
    std::uniform_int_distribution<size_t> indexDistribution; // Distribution for random index
};

#endif // DQN_H
