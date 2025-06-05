#ifndef LAYER_H
#define LAYER_H

class Layer {
public:
    virtual ~Layer() {}

    virtual void forward() = 0;
    virtual void backward() = 0;
};

#endif // LAYER_H