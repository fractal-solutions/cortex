#ifndef SIMPLEGRIDENVIRONMENT_H
#define SIMPLEGRIDENVIRONMENT_H

#include <vector>

class SimpleGridEnvironment {
public:
    SimpleGridEnvironment(int gridSize);
    std::vector<double> Reset();
    std::tuple<std::vector<double>, double, bool> Step(int action);

private:
    int gridSize;
    int currentPosition;
};

#endif // SIMPLEGRIDENVIRONMENT_H
