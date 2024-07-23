#ifndef EXPORT_H
#define EXPORT_H

#include "DQN.h"

// Export functions for the DLL
extern "C" {

__declspec(dllexport) DQN* __stdcall CreateDQN(int stateSize, int actionSize, int* hiddenLayers, int hiddenLayersSize);
__declspec(dllexport) void __stdcall DestroyDQN(DQN* dqn);
__declspec(dllexport) int __stdcall SelectAction(DQN* dqn, double* state, int stateSize, double epsilon);
__declspec(dllexport) void __stdcall Train(DQN* dqn, double* state, int stateSize, int action, double reward, double* nextState, int nextStateSize, double gamma, double epsilonDecay);
__declspec(dllexport) void __stdcall UpdateTargetNetwork(DQN* dqn);
__declspec(dllexport) double __stdcall GetGamma(DQN* dqn);
__declspec(dllexport) double __stdcall GetEpsilon(DQN* dqn);
__declspec(dllexport) double __stdcall GetEpsilonDecay(DQN* dqn);
__declspec(dllexport) int __stdcall SaveModel(DQN* dqn, const char* filepath);
__declspec(dllexport) int __stdcall LoadModel(DQN* dqn, const char* filepath);

}

#endif // EXPORT_H
