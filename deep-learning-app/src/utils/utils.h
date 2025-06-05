#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <string>

// Function to load data from a file
std::vector<std::vector<float>> loadData(const std::string& filePath);

// Function to preprocess data
std::vector<std::vector<float>> preprocessData(const std::vector<std::vector<float>>& data);

// Function to measure the performance of a model
void measurePerformance(const std::string& modelName, double duration, double accuracy);

#endif // UTILS_H