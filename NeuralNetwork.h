#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <vector>
#include <functional>

class NeuralNetwork {
public:
    NeuralNetwork(int inputSize, const std::vector<int>& hiddenLayers, int outputSize,
                  std::function<double(double)> activation = [](double x) { return x > 0 ? x : 0; }, // Default ReLU
                  std::function<double(double)> activationDerivative = [](double x) { return x > 0 ? 1 : 0; }); // Default ReLU derivative

    std::vector<double> Forward(const std::vector<double>& inputs);
    void Backward(const std::vector<double>& inputs, const std::vector<double>& targets, double learningRate);
    void UpdateWeights(const std::string& optimizer); // This method will be used in case of optimization or weight update.
    double CalculateLoss(const std::vector<double>& targets, const std::vector<double>& outputs);
    double GetLoss() const;
    
    
    //priiiivaaaaatee
    std::vector<int> layerSizes;
    std::vector<std::vector<std::vector<double>>> weights;
    std::vector<std::vector<double>> biases;
    std::vector<std::vector<std::vector<double>>> weightGradients;
    std::vector<std::vector<double>> biasGradients;
    std::function<double(double)> activation;
    std::function<double(double)> activationDerivative;
    double learningRate = 0.001; // Default learning rate
    double loss;


private:
    // Adam optimizer parameters
    std::vector<std::vector<std::vector<double>>> m_weights;
    std::vector<std::vector<std::vector<double>>> v_weights;
    std::vector<std::vector<double>> m_biases;
    std::vector<std::vector<double>> v_biases;
    double beta1 = 0.9;
    double beta2 = 0.999;
    double epsilon = 1e-8;
    int t = 0;

    void InitializeAdam(); // Initialize Adam parameters
    void UpdateWeightsSGD();
    void UpdateWeightsAdam();
    
};

#endif // NEURALNETWORK_H
