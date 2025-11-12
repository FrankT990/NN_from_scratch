#pragma once

#include <vector>
#include <string>
#include <cmath>
#include <stdexcept>
#include <algorithm> 

// Activation functions / derivtives
inline double relu(double x) { 
  return std::max(0.0, x); 
}
inline double relu_deriv(double x) { 
  return x > 0 ? 1.0 : 0.0;
}
inline double sig(double x) { 
  return 1.0 / (1.0 + std::exp(-x)); 
}
inline double sig_prime(double x) { 
  double s = sig(x); 
  return s * (1 - s); 
}
inline double Id(double x) { 
  return x; 
}

// softmax
inline std::vector<double> softmax(const std::vector<double>& output_layer) {
    std::vector<double> result(output_layer.size());
    double sum = 0.0;
    double maximum = 0;
    for (double num : output_layer)
      maximum = std::max(maximum, num);
    for (double val : output_layer) {
      sum += std::exp(val - maximum);
    }
    for (int i = 0; i < output_layer.size(); i++) 
      result[i] = std::exp(output_layer[i] - maximum) / sum;
    
    return result;
}
	
struct Layer{
  int size; // number of nodes
  int prev_layer_size; // number of nodes in previous layer
  std::string activation; // activation label
  std::vector<std::vector<double> > weights; // weights[i][j] corresponds to the weight to multiply synapse j from previous layer to node i;
  std::vector<double> biases; // biases for each node

  // for backpropagation
  std::vector<double> weighted_sum; 
  std::vector<double> activated_output;
  std::vector<std::vector<double> > dW; // gradients of weights
  std::vector<double> db; // gradients of biases

	Layer(int size, int prev_size, std::string activation);
  
};

class NeuralNetwork {
private:    
	int num_layers;
	std::vector<Layer*> layers;
public:
    NeuralNetwork(const std::vector<int> &node_counts, const std::vector<std::string> &activation_functions);
    ~NeuralNetwork();

    std::vector<double> forward_pass(const std::vector<double> &input);
    void backpropagation(const std::vector<double>& y_true, const std::vector<double>& y_pred);
    void update_parameters(double learning_rate, int batch_size);
    void reset_gradients();
    bool save_model(const std::string &filename); 
    bool load_model(const std::string &filename);
};