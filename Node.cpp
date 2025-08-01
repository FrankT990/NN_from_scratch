#include "node.h"
#include "accessory_functions.h"
#include <vector>
#include <functional>
#include <cmath>

class Node{

  public:
    Node(int input_size = 0){
      // Initialize weights and bias randomly
      for (int i=0; i < input_size; i++)
        this->weights.push_back(random_num_generator(-1.0, 1.0));
      this->bias = random_num_generator(-1.0, 1.0);

      // default activation function 
      this->activation = reLU;
      this->data = 0.0;
    }

    Node(int input_size, std::function<double(double)> activation_function){
      // Initialize weights and bias randomly
      for (int i=0; i < input_size; i++)
        this->weights.push_back(random_num_generator(-1.0, 1.0));
      this->bias = random_num_generator(-1.0, 1.0);

      // default activation function 
      this->activation = activation_function;
      this->data = 0.0;
    }

  private:
    double data;
    double bias;
    std::vector<double> weights;
    std::function<double(double)> activation;

}