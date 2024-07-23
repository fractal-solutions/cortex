#include "DQN.h"
#include <vector>

extern "C" {

DQN* __stdcall CreateDQN(int stateSize, int actionSize, int* hiddenLayers, int hiddenLayersSize) {
    std::vector<int> hiddenLayersVec(hiddenLayers, hiddenLayers + hiddenLayersSize);
    return new DQN(stateSize, actionSize, hiddenLayersVec);
}

void __stdcall DestroyDQN(DQN* dqn) {
    delete dqn;
}

int __stdcall SelectAction(DQN* dqn, double* state, int stateSize, double epsilon) {
    std::vector<double> stateVec(state, state + stateSize);
    return dqn->SelectAction(stateVec, epsilon);
}

void __stdcall Train(DQN* dqn, double* state, int stateSize, int action, double reward, double* nextState, int nextStateSize, double gamma, double epsilonDecay) {
    std::vector<double> stateVec(state, state + stateSize);
    std::vector<double> nextStateVec(nextState, nextState + nextStateSize);
    dqn->Train(stateVec, action, reward, nextStateVec, gamma, epsilonDecay);
}

void __stdcall UpdateTargetNetwork(DQN* dqn) {
    dqn->UpdateTargetNetwork();
}

double __stdcall GetGamma(DQN* dqn) {
    return dqn->GetGamma();
}

double __stdcall GetEpsilon(DQN* dqn) {
    return dqn->GetEpsilon();
}

double __stdcall GetEpsilonDecay(DQN* dqn) {
    return dqn->GetEpsilonDecay();
}

}
