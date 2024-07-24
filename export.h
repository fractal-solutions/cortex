#ifndef EXPORT_H
#define EXPORT_H

#include "DQN.h"

// Define cross-platform export macros
#if defined(_WIN32) || defined(_WIN64)
    #define DLL_EXPORT __declspec(dllexport)
    #define DLL_CALL __stdcall
#else
    #define DLL_EXPORT __attribute__((visibility("default")))
    #define DLL_CALL
#endif

// Export functions for the DLL
extern "C" {

DLL_EXPORT DQN* DLL_CALL CreateDQN(int stateSize, int actionSize, int* hiddenLayers, int hiddenLayersSize);
DLL_EXPORT void DLL_CALL DestroyDQN(DQN* dqn);
DLL_EXPORT int DLL_CALL SelectAction(DQN* dqn, double* state, int stateSize, double epsilon);
DLL_EXPORT void DLL_CALL Train(DQN* dqn, double* state, int stateSize, int action, double reward, double* nextState, int nextStateSize, double gamma, double epsilonDecay);
DLL_EXPORT void DLL_CALL UpdateTargetNetwork(DQN* dqn);
DLL_EXPORT double DLL_CALL GetGamma(DQN* dqn);
DLL_EXPORT double DLL_CALL GetEpsilon(DQN* dqn);
DLL_EXPORT double DLL_CALL GetEpsilonDecay(DQN* dqn);
DLL_EXPORT int DLL_CALL SaveModel(DQN* dqn, const char* filepath);
DLL_EXPORT int DLL_CALL LoadModel(DQN* dqn, const char* filepath);

}

#endif // EXPORT_H
