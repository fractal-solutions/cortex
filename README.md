# DQN LIBRARY FOR MQL5
Cortex is a cutting-edge Deep Q-Network (DQN) library that's meticulously crafted for seamless integration with MetaTrader 5 (MQL5) 🚀. 

It empowers traders to harness the advanced capabilities of reinforcement learning, taking their trading strategies to the next level 🚀📈. 

By tapping into the power of deep Q-networks, Cortex enables the creation of sophisticated AI-driven decision-making processes that boost trading performance and maximize profits 💹💰.


## How To .....
The program files are compiled to a DLL which can be used in MQL5 by placing in `MQL5/Libraries` folder. 

Incase you want to edit the library for optimization purposes you are free to tinker just remember to build so that you get your DLL file.

### Build Command for DLL
`g++ -shared -o dqn.dll export.cpp DQN.cpp NeuralNetwork.cpp`

## Using the DLL in MQL5
In your MQL5 code, you'll need to import the functions from your DLL and call them. Here's how you can set it up:

#### Architecture Overview
1. Initialization (OnInit):
    > Create the DQN instance.
    > Optionally load a pre-trained model.

2. Tick Handling (OnTick):
    > Use the DQN to make decisions based on incoming market signals.
    > Optionally train the DQN based on actions taken and received rewards.

3. Deinitialization (OnDeinit):
    > Save the current model state.
    > Clean up resources.


### Example MQL5 Code:

```

// DLL IMPORT
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

// Global variables
int dqn;
double gamma;
double epsilon;
double epsilonDecay;
string modelPath = "DQN_Model.dat";

// Helper function to print error messages
void PrintError(int errorCode) {
    switch (errorCode) {
        case 0:
            Print("Operation successful.");
            break;
        case 1:
            Print("Error: Invalid parameters.");
            break;
        case 2:
            Print("Error: Failed to open file.");
            break;
        default:
            Print("Unknown error code: ", errorCode);
            break;
    }
}

// Initialization function
int OnInit() {
    int stateSize = 4;  // Example state size
    int actionSize = 2; // Example action size
    int hiddenLayers[] = {24, 24}; // Example hidden layers
    int hiddenLayersSize = ArraySize(hiddenLayers);

    // Create DQN instance
    dqn = CreateDQN(stateSize, actionSize, hiddenLayers, hiddenLayersSize);

    // Load model if available
    int loadResult = LoadModel(dqn, modelPath);
    PrintError(loadResult);

    // Get DQN parameters
    gamma = GetGamma(dqn);
    epsilon = GetEpsilon(dqn);
    epsilonDecay = GetEpsilonDecay(dqn);

    return INIT_SUCCEEDED;
}

// Deinitialization function
void OnDeinit(const int reason) {
    // Save the model before exiting
    int saveResult = SaveModel(dqn, modelPath);
    PrintError(saveResult);

    // Destroy the DQN instance
    DestroyDQN(dqn);
}

// Function to handle new ticks
void OnTick() {
    // Example state (replace with actual market data)
    double state[] = {iClose(Symbol(), PERIOD_M1, 1), iHigh(Symbol(), PERIOD_M1, 1), iLow(Symbol(), PERIOD_M1, 1), iVolume(Symbol(), PERIOD_M1, 1)};

    // Select an action
    int action = SelectAction(dqn, state, ArraySize(state), epsilon);

    // Execute the action (replace with actual trading logic)
    if (action == 0) {
        // Buy signal
        if (OrderSelect(0, SELECT_BY_POS)) {
            if (OrderType() == OP_SELL) OrderClose(OrderTicket(), OrderLots(), Ask, 2);
        }
        OrderSend(Symbol(), OP_BUY, 0.1, Ask, 2, 0, 0);
    } else {
        // Sell signal
        if (OrderSelect(0, SELECT_BY_POS)) {
            if (OrderType() == OP_BUY) OrderClose(OrderTicket(), OrderLots(), Bid, 2);
        }
        OrderSend(Symbol(), OP_SELL, 0.1, Bid, 2, 0, 0);
    }

    // Example reward (replace with actual reward calculation)
    double reward = OrderProfit();

    // Example next state (replace with actual market data)
    double nextState[] = {iClose(Symbol(), PERIOD_M1, 0), iHigh(Symbol(), PERIOD_M1, 0), iLow(Symbol(), PERIOD_M1, 0), iVolume(Symbol(), PERIOD_M1, 0)};

    // Train the model
    Train(dqn, state, ArraySize(state), action, reward, nextState, ArraySize(nextState), gamma, epsilonDecay);

    // Update the target network periodically
    if (TimeCurrent() % 60 == 0) {
        UpdateTargetNetwork(dqn);
    }
}


```

### 