#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <cmath>
#include <stdexcept>
#include <filesystem>
#include <utility>   
#include <algorithm>   
#include <iomanip> 

#include "Loader.h"
#include "NeuralNetwork.h"

#include "json.hpp"
using json = nlohmann::json;

int get_prediction_index(const std::vector<double>& y_pred) {
  return std::distance(y_pred.begin(), std::max_element(y_pred.begin(), y_pred.end()));
}


int main(int argc, char* argv[]) {
  json config;
  std::ifstream f("config.json");
  if (!f.is_open()) {
    std::cerr << "Error: Could not open config.json" << std::endl;
    return 1;
  }
  f >> config;
  std::string dataset_path = config["dataset"];
  std::vector<std::string> class_names = config["layers"].back()["classes"];

  std::vector<int> node_counts;
  std::vector<std::string> activation_functions;
  
  int input_layer_size = 0;
  std::vector<std::pair<std::vector<double>, std::vector<double> > > temp_data;
  temp_data = load_dataset(dataset_path, {class_names[0]}, input_layer_size); 
  if (input_layer_size <= 0) {
     std::cerr << "Error: Could not determine input size from dataset: " << dataset_path << std::endl;
     return 1;
  }
  temp_data.clear(); // Free the memory

  // make neural network
  node_counts.push_back(input_layer_size); 
  activation_functions.push_back("Id");
  for (const auto &layer_config : config["layers"]) {
    node_counts.push_back(layer_config["nodes"]);
    activation_functions.push_back(layer_config["activation"]);
  }

  NeuralNetwork nn(node_counts, activation_functions); // <-- Now this is correct

  if (argc == 2 && std::string(argv[1]) == "--train") {
    std::cout << "--- Starting Training ---" << std::endl;
    std::vector<std::pair<std::vector<double>, std::vector<double>>> training_data;
    training_data = load_dataset(dataset_path, class_names, input_layer_size);
    if (training_data.empty()) {
      std::cerr << "Error: Training data is empty." << std::endl;
      return 1;
    }

    int epochs = 10;
    double learning_rate = 0.01;
    int batch_size = 32;
    
    std::cout << "Epochs: " << epochs << ", LR: " << learning_rate << ", Batch Size: " << batch_size << std::endl;
    std::random_device rd;
    std::mt19937 g(rd());
    
    for (int i = 0; i < epochs; i++) {
      std::shuffle(training_data.begin(), training_data.end(), g);
      for (int j = 0; j < training_data.size(); j += batch_size) {
        nn.reset_gradients(); 
        int current_batch_size = std::min(batch_size, (int)training_data.size() - j);
        for (int k = 0; k < current_batch_size; ++k) {
          const auto& s = training_data[j + k];
          std::vector<double> y_pred = nn.forward_pass(s.first);
          nn.backpropagation(s.second, y_pred);
        }
        nn.update_parameters(learning_rate, current_batch_size);
      }
      std::cout << "Epoch " << (i + 1) << "/" << epochs << " completed." << std::endl;
    }
    
    // save model
    if (nn.save_model("model.json")) {
      std::cout << "Training complete. Model saved to model.json" << std::endl;
    } else {
      std::cerr << "Error: Training complete, but failed to save model." << std::endl;
    }

  // testing
  } else if (argc == 3 && std::string(argv[1]) == "--test_folder") {
    std::cout << "--- Starting Testing ---" << std::endl;
    if (!nn.load_model("model.json")) {
      std::cerr << "Error: Could not load model.json. Have you trained the network first?" << std::endl;
      return 1;
    }

    std::string test_path = argv[2];
    int test_input_size = 0;
    std::vector<std::pair<std::vector<double>, std::vector<double>>> test_data;
    test_data = load_dataset(test_path, class_names, test_input_size);

    if (test_data.empty()) {
      std::cerr << "Error: No test data loaded from " << test_path << std::endl;
      return 1;
    }
    if (test_input_size != input_layer_size) {
      std::cerr << "Error: Test data size (" << test_input_size << ") does not match train data size (" << input_layer_size << ")" << std::endl;
      return 1;
    }

    int correct_predictions = 0;
    for (const auto& s : test_data) {
      std::vector<double> y_pred = nn.forward_pass(s.first);
      int pred_index = get_prediction_index(y_pred);
      int true_index = get_prediction_index(s.second); // Also works for one-hot

      if (pred_index == true_index) {
        correct_predictions++;
      }
    }
    
    double accuracy = static_cast<double>(correct_predictions) / test_data.size();
    std::cout << "--- Test Results ---" << std::endl;
    std::cout << "Total Samples: " << test_data.size() << std::endl;
    std::cout << "Correct: " << correct_predictions << std::endl;
    std::cout << "Accuracy: " << std::fixed << std::setprecision(2) << (accuracy * 100.0) << "%" << std::endl;

  // single image prediction
  } else if (argc == 3 && std::string(argv[1]) == "--predict") {
    
    std::string image_path = argv[2];
    std::vector<double> image_data = load_single_image(image_path, input_layer_size);

    if (image_data.empty()) {
      std::cerr << "Error: Could not load or process image." << std::endl;
      return 1;
    }
    
    std::vector<double> y_pred = nn.forward_pass(image_data);
    int pred_index = get_prediction_index(y_pred);

    std::cout << "--- Prediction ---" << std::endl;
    std::cout << "Prediction: " << class_names[pred_index] << std::endl;
    std::cout << "Confidence: " << std::fixed << std::setprecision(2) << (y_pred[pred_index] * 100.0) << "%" << std::endl;
    std::cout << "---" << std::endl;
    std::cout << "Full Output Vector:" << std::endl;
    for(int i = 0; i < class_names.size(); ++i) {
      std::cout << class_names[i] << ": " << std::fixed << std::setprecision(4) << y_pred[i] << std::endl;
    }

  } else {
    std::cerr << "Invalid usage." << std::endl;
    std::cerr << "Usage:" << std::endl;
    std::cerr << "  ./train_app          (To train the network)" << std::endl;
    std::cerr << "  ./train_app --test_folder <path>  (To test on a folder)" << std::endl;
    std::cerr << "  ./train_app --predict <path>    (To predict a single image)" << std::endl;
    return 1;
  }
  
  return 0;
}