#include "DQN.h"
#include "export.h"
#include <vector>

extern "C" {

DQN* DLL_CALL CreateDQN(int stateSize, int actionSize, int* hiddenLayers, int hiddenLayersSize) {
    std::vector<int> hiddenLayersVec(hiddenLayers, hiddenLayers + hiddenLayersSize);
    return new DQN(stateSize, actionSize, hiddenLayersVec);
}

void DLL_CALL DestroyDQN(DQN* dqn) {
    delete dqn;
}

int DLL_CALL SelectAction(DQN* dqn, double* state, int stateSize, double epsilon) {
    std::vector<double> stateVec(state, state + stateSize);
    return dqn->SelectAction(stateVec, epsilon);
}

void DLL_CALL Train(DQN* dqn, double* state, int stateSize, int action, double reward, double* nextState, int nextStateSize, double gamma, double epsilonDecay) {
    std::vector<double> stateVec(state, state + stateSize);
    std::vector<double> nextStateVec(nextState, nextState + nextStateSize);
    dqn->Train(stateVec, action, reward, nextStateVec, gamma, epsilonDecay);
}

void DLL_CALL UpdateTargetNetwork(DQN* dqn) {
    dqn->UpdateTargetNetwork();
}

double DLL_CALL GetGamma(DQN* dqn) {
    return dqn->GetGamma();
}

double DLL_CALL GetEpsilon(DQN* dqn) {
    return dqn->GetEpsilon();
}

double DLL_CALL GetEpsilonDecay(DQN* dqn) {
    return dqn->GetEpsilonDecay();
}

double DLL_CALL GetQNetLoss(DQN* dqn){
    return dqn->GetQNetLoss();
}

double DLL_CALL GetTargetNetLoss(DQN* dqn){
    return dqn->GetTargetNetLoss();
}

int DLL_CALL SaveModel(DQN* dqn, const char* filepath) {
    dqn->SaveModel(filepath);
}

int DLL_CALL LoadModel(DQN* dqn, const char* filepath) {
    dqn->LoadModel(filepath);
}

}
