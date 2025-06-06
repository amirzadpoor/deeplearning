#include <iostream>
#include "tensor.h"
#include "neuron.h"

int main() {
    std::cout << "Initializing Deep Learning Application..." << std::endl;

    // Create input tensor with 3 elements
    Tensor input({3});
    input({0}) = 1.0f;
    input({1}) = 2.0f;
    input({2}) = 3.0f;

    // Create neuron with 3 inputs
    LinearNeuron neuron(3);
    neuron.setWeights({0.5f, -1.0f, 2.0f});
    neuron.setBias(0.1f);

    float output = neuron.forward(input);
    std::cout << "Neuron output: " << output << std::endl; // Should print 1*0.5 + 2*(-1.0) + 3*2.0 + 0.1 = 0.5 - 2 + 6 + 0.1 = 4.6

    std::cout << "Deep Learning Application finished." << std::endl;
    return 0;
}