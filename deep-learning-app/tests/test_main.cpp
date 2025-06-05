#include <gtest/gtest.h>
#include "tensor.h"
#include "layer.h"
#include "model.h"

TEST(TensorTest, Addition) {
    Tensor a({2, 2});
    Tensor b({2, 2});
    a.fill(1.0);
    b.fill(2.0);
    Tensor result = a + b;
    EXPECT_EQ(result.get(0, 0), 3.0);
    EXPECT_EQ(result.get(1, 1), 3.0);
}

TEST(TensorTest, Multiplication) {
    Tensor a({2, 2});
    Tensor b({2, 2});
    a.fill(2.0);
    b.fill(3.0);
    Tensor result = a * b;
    EXPECT_EQ(result.get(0, 0), 6.0);
    EXPECT_EQ(result.get(1, 1), 6.0);
}

TEST(LayerTest, ForwardBackward) {
    Layer layer;
    Tensor input({2, 2});
    input.fill(1.0);
    Tensor output = layer.forward(input);
    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 2);
    
    Tensor grad_output({2, 2});
    grad_output.fill(0.5);
    Tensor grad_input = layer.backward(grad_output);
    EXPECT_EQ(grad_input.shape()[0], 2);
    EXPECT_EQ(grad_input.shape()[1], 2);
}

TEST(ModelTest, Train) {
    Model model;
    model.addLayer(new Layer());
    Tensor data({2, 2});
    data.fill(1.0);
    Tensor labels({2, 1});
    labels.fill(0.0);
    model.train(data, labels);
    EXPECT_TRUE(true); // Placeholder for actual training validation
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}