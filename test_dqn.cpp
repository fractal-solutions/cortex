#include <iostream>
#include "DQN.h"
#include "SimpleGridEnvironment.h"

void TestDQN() {
    // Initialize environment
    SimpleGridEnvironment env(10); // Grid size of 10

    // Initialize DQN
    std::vector<int> hiddenLayers = {2, 2};
    DQN dqn(1, 2, hiddenLayers); // State size of 1, action size of 2

    // Training loop
    for (int episode = 0; episode < 1000; ++episode) {
        auto state = env.Reset();
        bool done = false;
        double totalReward = 0.0;

        while (!done) {
            int action = dqn.SelectAction(state, dqn.GetEpsilon());
            
            // Traditional tuple unpacking
            std::tuple<std::vector<double>, double, bool> result = env.Step(action);
            std::vector<double> nextState = std::get<0>(result);
            double reward = std::get<1>(result);
            bool doneFlag = std::get<2>(result);

            done = doneFlag;
            totalReward += reward;

            dqn.Train(state, action, reward, nextState, dqn.GetGamma(), dqn.GetEpsilonDecay());
            state = nextState;
        }

        // Update target network
        dqn.UpdateTargetNetwork();

        std::cout << "Episode " << episode + 1 << ": Total Reward = " << totalReward << std::endl;
    }
}

int main() {
    TestDQN();
    return 0;
}
