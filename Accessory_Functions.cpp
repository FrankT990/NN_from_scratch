#include "accessory_functions.h"
#include <iostream>
#include <random>

double random_num_generator(double min, double max){
  std::random_device rd;  
  std::mt19937 gen(rd()); // Mersenne Twister RNG
  std::uniform_real_distribution<> dist(min, max);

  double random_number = dist(gen);
  return random_number;
}

double reLU(double x) {
    return (x > 0) ? x : 0;
}