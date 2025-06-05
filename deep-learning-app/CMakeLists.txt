cmake_minimum_required(VERSION 3.10)

project(DeepLearningApp)

set(CMAKE_CXX_STANDARD 14)

include_directories(include)

file(GLOB SOURCES
    src/core/tensor.cpp
    src/core/neuron.cpp
    src/layers/layer.cpp
    src/models/model.cpp
    src/main.cpp
)

add_executable(deep_learning_app ${SOURCES})

enable_testing()

file(GLOB TEST_SOURCES tests/test_main.cpp)
add_executable(DeepLearningAppTests ${TEST_SOURCES})
target_link_libraries(DeepLearningAppTests gtest gtest_main)
add_test(NAME DeepLearningAppTests COMMAND DeepLearningAppTests)