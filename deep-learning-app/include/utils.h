#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <vector>

// Function to load data from a file
std::vector<std::vector<float>> loadData(const std::string& filename);

// Function to preprocess data
std::vector<std::vector<float>> preprocessData(const std::vector<std::vector<float>>& data);

// Function to measure performance
void measurePerformance(const std::string& operation, double duration);

#endif // UTILS_H