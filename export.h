#ifndef EXPORT_H
#define EXPORT_H

#ifdef __cplusplus
extern "C" {
#endif

// Function to create a DQN instance
__declspec(dllexport) void* CreateDQN(int stateSize, int actionSize, int hiddenLayers[], int hiddenLayersSize);

// Function to select an action given the current state
__declspec(dllexport) int SelectAction(void* dqn, double state[], int stateSize, double epsilon);

// Function to train the DQN with a given experience
__declspec(dllexport) void Train(void* dqn, double state[], int stateSize, int action, double reward, double nextState[], int nextStateSize, double gamma, double epsilonDecay);

// Function to update the target network
__declspec(dllexport) void UpdateTargetNetwork(void* dqn);

// Function to destroy the DQN instance
__declspec(dllexport) void DestroyDQN(void* dqn);

// Function to 
__declspec(dllexport) double GetEpsilon(void* dqn);

// Function to 
__declspec(dllexport) double GetGamma(void* dqn);

// Function to 
__declspec(dllexport) double GetEpsilonDecay(void* dqn);

#ifdef __cplusplus
}
#endif

#endif // EXPORT_H
