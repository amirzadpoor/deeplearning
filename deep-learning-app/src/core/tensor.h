class Tensor {
public:
    Tensor(int dimensions, const std::vector<int>& shape);
    Tensor(const Tensor& other);
    Tensor& operator=(const Tensor& other);
    ~Tensor();

    void reshape(const std::vector<int>& new_shape);
    Tensor operator+(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;

    float& operator()(const std::vector<int>& indices);
    const float& operator()(const std::vector<int>& indices) const;

    std::vector<int> getShape() const;
    int getDimensions() const;

private:
    int dimensions;
    std::vector<int> shape;
    std::vector<float> data;

    void allocateData();
    int calculateIndex(const std::vector<int>& indices) const;
};