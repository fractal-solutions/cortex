#include "SimpleGridEnvironment.h"
#include <tuple>
#include <vector>

SimpleGridEnvironment::SimpleGridEnvironment(int gridSize) : gridSize(gridSize), currentPosition(0) {}

std::vector<double> SimpleGridEnvironment::Reset() {
    currentPosition = 0;
    return std::vector<double>(1, static_cast<double>(currentPosition));
}

std::tuple<std::vector<double>, double, bool> SimpleGridEnvironment::Step(int action) {
    currentPosition += action;
    bool done = currentPosition >= gridSize;
    double reward = done ? 1.0 : -0.1;
    return {std::vector<double>(1, static_cast<double>(currentPosition)), reward, done};
}
