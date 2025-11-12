#include "NeuralNetwork.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <unordered_map>
#include <cmath>
#include <stdexcept>
#include "json.hpp"
using json = nlohmann::json;

Layer::Layer(int size, int prev_layer_size, std::string activation)
    : size(size), prev_layer_size(prev_layer_size), activation(activation){
  
  if(prev_layer_size > 0){ // not input layer
    weighted_sum.resize(size);
    activated_output.resize(size);
    db.resize(size, 0.0);
    dW.resize(size, std::vector<double>(prev_layer_size, 0.0));

    std::random_device rd;
    std::mt19937 rand_engine(rd());
    std::normal_distribution<double> dist(0.0, std::sqrt(2.0 / prev_layer_size));

    biases.resize(size, 0.01);
    weights.resize(size);

    for (int i = 0; i < size; i++) {
      weights[i].resize(prev_layer_size);
      for (int j = 0; j < prev_layer_size; j++) {
        weights[i][j] = dist(rand_engine); 
      }
    }
    } else {
    activated_output.resize(size);
  }
}

// node_counts includes input layer size
NeuralNetwork::NeuralNetwork(const std::vector<int> &node_counts, const std::vector<std::string> &activation_functions){
  // Add layers
  num_layers = node_counts.size();
  for (int i = 0; i < num_layers; i++){
    int prev_size = (i == 0)? 0 : node_counts[i - 1];
    layers.push_back(new Layer(node_counts[i], prev_size, activation_functions[i]));
  }
  
  // Initialize weights and biases for each layer
  // weights initialized randomly
  // biases set to 0 by default
  for (int i = 1; i < num_layers; i++){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(0, 1);
    int curr_layer_size = node_counts[i];
    int prev_layer_size = node_counts[i - 1];

    Layer* layer = layers[i];
    for (int j = 0; j < curr_layer_size; j++){
      std::vector<double> weights_for_node_j;
      for (int k = 0; k < prev_layer_size; k++){
        weights_for_node_j.push_back(d(gen) * 0.01); 
      }
      layer->weights.push_back(weights_for_node_j);
      layer->biases.push_back(0.0);
    }
  }
}

std::vector<double> NeuralNetwork::forward_pass(const std::vector<double> &input){
  layers[0]->activated_output = input;
  
  // for each layer
  for (int i = 1; i < num_layers; i++){

    std::vector<double> &prev = layers[i-1]->activated_output;
    Layer* layer = layers[i];

    // for each node in layer
    for (int j = 0; j < layer->size; j++){
      double sum = layer->biases[j];
      // for each synapse from previous layer into current node
      for (int k = 0; k < prev.size(); k++)
        sum += layer->weights[j][k] * prev[k];
      layer->weighted_sum[j] = sum;
      // apply activation
      if (layer->activation == "relu")
        layer->activated_output[j] = relu(sum);  
      else if (layer->activation == "sigmoid")
        layer->activated_output[j] = sig(sum);
      else if (layer->activation == "Id")
        layer->activated_output[j] = Id(sum);
    }
    if (layer->activation == "softmax"){
      layer->activated_output = softmax(layer->weighted_sum);
    }
    
  }
  // return output layer activations
  return layers.back()->activated_output; 
}

void NeuralNetwork::backpropagation(const std::vector<double> &y_pred, const std::vector<double> &y_actual){
  std::vector<double> delta;
  Layer* output_layer = layers.back();

  // cross entropy loss delta
  for (int i = 0; i < output_layer->size; i++)
    delta.push_back(y_pred[i] - y_actual[i]); 
  
  for (int i = layers.size()-1; i >= 1; i--){
    Layer* layer = layers[i];
    Layer* prev_layer = layers[i-1];

    // compute gradients for each layer
    for (int j = 0; j < layer->size; j++){
      // weights
      for (int k = 0; k < prev_layer->size; k++){
        // gradient of weight with contribution from node k in previous layer to node j in current layer
        layer->dW[j][k] += delta[j] * prev_layer->activated_output[k];
      }
      // bias
      layer->db[j] += delta[j];
    }

    // compute delta for previous layer
    if (i > 1){
      std::vector<double> new_delta(prev_layer->size, 0.0);
      for (int j = 0; j < prev_layer->size; j++){
        double sum = 0.0;
        for (int k = 0; k < layer->size; k++)
          sum += layer->weights[k][j] * delta[k];
        
        if (prev_layer->activation == "relu")
          new_delta[j] = sum * relu_deriv(prev_layer->weighted_sum[j]);
        else if (prev_layer->activation == "sigmoid")
          new_delta[j] = sum * sig_prime(prev_layer->weighted_sum[j]);
        else if (prev_layer->activation == "Id")
          new_delta[j] = sum; 
      }
      delta = new_delta;
    }
  }
}

void NeuralNetwork::update_parameters(double learning_rate, int batch_size){
  double clip_threshold = 1.0;

  if (batch_size == 0) return;

  for (int l = 1; l < num_layers; ++l) {
    Layer* layer = layers[l];
    for (int i = 0; i < layer->size; ++i) {
      double grad_b = layer->db[i] / batch_size;
      // make sure grads behave nice
      if (std::isnan(grad_b) || std::isinf(grad_b)) {
        grad_b = 0.0;
      }
      if (grad_b > clip_threshold) {
        grad_b = clip_threshold;
      } else if (grad_b < -clip_threshold) {
        grad_b = -clip_threshold;
      }
      
    
      layer->biases[i] -= learning_rate * grad_b;
      for (int j = 0; j < layer->prev_layer_size; ++j) {
        double grad_w = layer->dW[i][j] / batch_size;

        if (std::isnan(grad_w) || std::isinf(grad_w)) {
          grad_w = 0.0;
        }
        if (grad_w > clip_threshold) {
          grad_w = clip_threshold;
        } else if (grad_w < -clip_threshold) {
          grad_w = -clip_threshold;
        }
        layer->weights[i][j] -= learning_rate * grad_w;
        layer->dW[i][j] = 0.0;
      }

      // Reset bias gradient for next sample
      layer->db[i] = 0.0;
    }
  }
}

void NeuralNetwork::reset_gradients() {
  for (int i = 1; i < num_layers; i++) {
    Layer* layer = layers[i];
    for (auto &row : layer->dW) {
      std::fill(row.begin(), row.end(), 0.0);
    }
    std::fill(layer->db.begin(), layer->db.end(), 0.0);
  }
}
NeuralNetwork::~NeuralNetwork(){
  for (auto layer : layers){
    delete layer;
  }
}

// file saving and loading - uses ai generated code
bool NeuralNetwork::save_model(const std::string& filename) {
  nlohmann::json model_json;
  model_json["num_layers"] = num_layers;
  
  nlohmann::json layers_json = nlohmann::json::array();

  for (int i = 1; i < num_layers; i++) { 
    nlohmann::json layer_json;

    nlohmann::json biases_json = nlohmann::json::array();
    for (double bias : layers[i]->biases) {
      biases_json.push_back(bias);
    }
    layer_json["biases"] = biases_json;

    nlohmann::json weights_json = nlohmann::json::array();
    for (const auto& weights_for_node : layers[i]->weights) {
      nlohmann::json weights_node_json = nlohmann::json::array();
      for (double weight : weights_for_node) {
        weights_node_json.push_back(weight);
      }
      weights_json.push_back(weights_node_json);
    }
    layer_json["weights"] = weights_json;
    
    layers_json.push_back(layer_json);
  }
  
  model_json["layers"] = layers_json;

  std::ofstream f(filename);
  if (!f.is_open()) {
    std::cerr << "Error: Could not open " << filename << " for writing." << std::endl;
    return false;
  }
  f << model_json.dump(2); 
  f.close();
  return true;
}

bool NeuralNetwork::load_model(const std::string& filename) {
  nlohmann::json model_json;
  std::ifstream f(filename);
  if (!f.is_open()) {
    std::cerr << "Error: Could not open " << filename << " for reading." << std::endl;
    return false;
  }
  try {
    f >> model_json;
  } catch (nlohmann::json::parse_error& e) {
    std::cerr << "Error parsing model file: " << e.what() << std::endl;
    return false;
  }

  try {
    nlohmann::json json_layers = model_json["layers"];
    if (json_layers.size() != this->num_layers - 1) {
        std::cerr << "Error: Model file layer count (" << json_layers.size() 
                  << ") does not match network layer count (" << (this->num_layers - 1) << ")." << std::endl;
        return false;
    }

    for (int i = 1; i < this->num_layers; ++i) {
      Layer* current_layer = this->layers[i];
      nlohmann::json current_json_layer = json_layers[i - 1]; 

      nlohmann::json biases_json = current_json_layer["biases"];
      if (biases_json.size() != current_layer->size) {
        std::cerr << "Error: Bias count mismatch in layer " << i << std::endl;
        return false;
      }
      for (int j = 0; j < current_layer->size; ++j) {
        if (biases_json[j].is_null()) {
          std::cerr << "Error: 'null' value encountered loading bias for layer " << i << ", node " << j << std::endl;
          return false;
        }
        current_layer->biases[j] = biases_json[j];
      }

      nlohmann::json weights_json = current_json_layer["weights"];
      if (weights_json.size() != current_layer->size) {
        std::cerr << "Error: Weight node count mismatch in layer " << i << std::endl;
        return false;
      }
      for (int j = 0; j < current_layer->size; ++j) {
        nlohmann::json weights_node_json = weights_json[j];
        if (weights_node_json.size() != current_layer->prev_layer_size) {
          std::cerr << "Error: Weight connection count mismatch in layer " << i << ", node " << j << std::endl;
          return false;
        }
        for (int k = 0; k < current_layer->prev_layer_size; ++k) {
          if (weights_node_json[k].is_null()) {
            std::cerr << "Error: 'null' value encountered loading weight for layer " << i << ", node " << j << ", connection " << k << std::endl;
            return false;
          }
          current_layer->weights[j][k] = weights_node_json[k];
        }
      }
    }
  } catch (nlohmann::json::exception& e) {
    std::cerr << "Error processing model JSON data: " << e.what() << std::endl;
    return false;
  }

  f.close(); 
  std::cout << "Model loaded successfully from " << filename << std::endl;
  return true;
}


