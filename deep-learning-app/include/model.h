#ifndef MODEL_H
#define MODEL_H

#include <vector>
#include "layer.h"

class Model {
public:
    Model();
    ~Model();

    void addLayer(Layer* layer);
    void compile();
    void fit(const Tensor& x, const Tensor& y, int epochs, int batchSize);
    Tensor predict(const Tensor& x);

private:
    std::vector<Layer*> layers;
};

#endif // MODEL_H