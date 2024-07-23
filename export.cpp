#include "export.h"
#include "DQN.h"

extern "C" {

__declspec(dllexport) void* CreateDQN(int stateSize, int actionSize, int hiddenLayers[], int hiddenLayersSize) {
    std::vector<int> layers(hiddenLayers, hiddenLayers + hiddenLayersSize);
    return new DQN(stateSize, actionSize, layers);
}

__declspec(dllexport) int SelectAction(void* dqn, double state[], int stateSize, double epsilon) {
    std::vector<double> stateVec(state, state + stateSize);
    return static_cast<DQN*>(dqn)->SelectAction(stateVec, epsilon);
}

__declspec(dllexport) void Train(void* dqn, double state[], int stateSize, int action, double reward, double nextState[], int nextStateSize, double gamma, double epsilonDecay) {
    std::vector<double> stateVec(state, state + stateSize);
    std::vector<double> nextStateVec(nextState, nextState + nextStateSize);
    static_cast<DQN*>(dqn)->Train(stateVec, action, reward, nextStateVec, gamma, epsilonDecay);
}

__declspec(dllexport) void UpdateTargetNetwork(void* dqn) {
    static_cast<DQN*>(dqn)->UpdateTargetNetwork();
}

__declspec(dllexport) double GetEpsilon(void* dqn) {
    return static_cast<DQN*>(dqn)->GetEpsilon();
}

__declspec(dllexport) double GetEpsilonDecay(void* dqn) {
    return static_cast<DQN*>(dqn)->GetEpsilonDecay();
}

__declspec(dllexport) double GetGamma(void* dqn) {
    return static_cast<DQN*>(dqn)->GetGamma();
}

__declspec(dllexport) void DestroyDQN(void* dqn) {
    delete static_cast<DQN*>(dqn);
}

}
