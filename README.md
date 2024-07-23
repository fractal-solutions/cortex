# DQN LIBRARY FOR MQL5
## Build Command for DLL
`g++ -shared -o dqn.dll export.cpp DQN.cpp NeuralNetwork.cpp`

## Using the DLL in MQL5
In your MQL5 code, you'll need to import the functions from your DLL and call them. Here's how you can set it up:
### Example MQL5 Code:

```

#import "dqn.dll"
int CreateDQN(int stateSize, int actionSize, int &hiddenLayers[], int hiddenLayersSize);
int SelectAction(int dqn, double &state[], int stateSize, double epsilon);
void Train(int dqn, double &state[], int stateSize, int action, double reward, double &nextState[], int nextStateSize, double gamma, double epsilonDecay);
void UpdateTargetNetwork(int dqn);
void DestroyDQN(int dqn);
double GetGamma(int dqn);
double GetEpsilon(int dqn);
double GetEpsilonDecay(int dqn);
int SaveModel(int dqn, string filepath);
int LoadModel(int dqn, string filepath);
#import

// Example usage
void OnStart() {
    int stateSize = 4;
    int actionSize = 2;
    int hiddenLayers[] = {24, 24};
    int hiddenLayersSize = ArraySize(hiddenLayers);

    // Create DQN instance
    int dqn = CreateDQN(stateSize, actionSize, hiddenLayers, hiddenLayersSize);

    // Load model if exists
    int loadError = LoadModel(dqn, modelPath);
    if (loadError == 0) {
        Comment("Model Loaded")
    } else {
        Comment("Error: No File To Load")
    }

    // Example state
    double state[] = {0.1, 0.2, 0.3, 0.4};
    double epsilon = 1.0;

    // Select an action
    int action = SelectAction(dqn, state, ArraySize(state), GetEpsilon(dqn));
    
    // Example training
    double reward = 1.0;
    double nextState[] = {0.2, 0.3, 0.4, 0.5};
    Train(dqn, state, ArraySize(state), action, reward, nextState, ArraySize(nextState), GetGamma(dqn), GetEpsilonDecay(dqn));

    // Update target network
    UpdateTargetNetwork(dqn);

    // Save model
    int saveError = SaveModel(dqn, modelPath);
    if (saveError == 0) {
        Comment("Model Saved")
    } else {
        Comment("Error: No File To Save")
    }

    // Destroy DQN instance
    DestroyDQN(dqn);
}


```