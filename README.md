# DQN LIBRARY FOR MQL5
## Build Command for DLL
`g++ -shared -o dqn.dll export.cpp DQN.cpp NeuralNetwork.cpp`

## Using the DLL in MQL5
In your MQL5 code, you'll need to import the functions from your DLL and call them. Here's how you can set it up:
### Example MQL5 Code:

```

#import "dqn.dll"
void* CreateDQN(int stateSize, int actionSize, int hiddenLayers[], int hiddenLayersSize);
int SelectAction(void* dqn, double state[], int stateSize, double epsilon);
void Train(void* dqn, double state[], int stateSize, int action, double reward, double nextState[], int nextStateSize, double gamma, double epsilonDecay);
void UpdateTargetNetwork(void* dqn);
void DestroyDQN(void* dqn);
double GetGamma(void* dqn);
double GetEpsilon(void* dqn);
double GetEpsilonDecay(void* dqn);
#import

// Example usage
void OnStart() {
    int stateSize = 4;
    int actionSize = 2;
    int hiddenLayers[] = {24, 24};
    int hiddenLayersSize = ArraySize(hiddenLayers);

    // Create DQN instance
    void* dqn = CreateDQN(stateSize, actionSize, hiddenLayers, hiddenLayersSize);

    // Example state
    double state[] = {0.1, 0.2, 0.3, 0.4};
    double epsilon = 1.0;

    // Select an action
    int action = SelectAction(dqn, state, stateSize, GetEpsilon(dqn));

    // Example training
    double reward = 1.0;
    double nextState[] = {0.2, 0.3, 0.4, 0.5};
    double gamma = 0.99;
    double epsilonDecay = 0.995;
    Train(dqn, state, stateSize, action, reward, nextState, stateSize, GetGamma(dqn), GetEpsilonDecay(dqn));

    // Update target network
    UpdateTargetNetwork(dqn);

    // Destroy DQN instance
    DestroyDQN(dqn);
}

```