class Layer {
public:
    virtual ~Layer() {}

    virtual void forward(const Tensor& input) = 0;
    virtual Tensor backward(const Tensor& gradient) = 0;
    virtual void updateWeights(double learningRate) = 0;
};