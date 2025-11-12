#include "Loader.h"


#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <iostream>
#include <filesystem>
#include <map>
#include <random>
#include <algorithm>
#include <iostream>

namespace fs = std::filesystem;

// note: some code for file handling is ai generated
std::vector<std::pair<std::vector<double>, std::vector<double> > >
load_dataset(const std::string& folder_path,
    const std::vector<std::string>& classes,
    int &input_size){
  std::vector<std::pair<std::vector<double>, std::vector<double> > > dataset;
  input_size = -1;

  std::map<std::string, int> class_to_index;
  for (int i = 0; i < classes.size(); ++i) {
    class_to_index[classes[i]] = i;
  }

  for (const auto  &class_label : classes) { 
    fs::path class_path = fs::path(folder_path) / class_label;
    int class_index = class_to_index[class_label];

    if (!fs::exists(class_path)) {
      std::cerr << "Warning: Folder does not exist: " << class_path << std::endl;
      continue;
    }

    for (const auto& entry : fs::directory_iterator(class_path)) {
      std::string image_path = entry.path().string();
      
      int width, height, channels;
      unsigned char* data = stbi_load(image_path.c_str(), &width, &height, &channels, 1);
      if (!data) {
        std::cerr << "Failed to load image: " << image_path << std::endl;
        continue;
      }

      if (input_size == -1) {
        input_size = width * height;

      } else if (input_size != width * height) {
        std::cerr << "Error: Image size mismatch! " << image_path << std::endl;
        stbi_image_free(data);
        continue;
      }

      std::pair<std::vector<double>, std::vector<double> > s;
      s.first.resize(input_size);
      for (int i = 0; i < input_size; i++) {
        s.first[i] = static_cast<double>(data[i]) / 255.0; 
      }
      stbi_image_free(data);

      s.second.resize(classes.size(), 0.0); 
      s.second[class_index] = 1.0;      
      dataset.push_back(s);
    }
  }
  std::cout << "Loaded " << dataset.size() << " samples." << std::endl;
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(dataset.begin(), dataset.end(), g);
  return dataset;
}

// for testing individual images
std::vector<double> load_single_image(const std::string& image_path, int expected_size) {
  int width, height, channels;
  unsigned char* data = stbi_load(image_path.c_str(), &width, &height, &channels, 1);
  if (!data) {
    std::cerr << "Failed to load image: " << image_path << std::endl;
    return {};
  }

  int image_size = width * height;
  if (image_size != expected_size) {
    std::cerr << "Error: Image size is " << image_size << ", but network expected " << expected_size << std::endl;
    stbi_image_free(data);
    return {};
  }

  std::vector<double> image_data(image_size);
  for (int i = 0; i < image_size; i++) 
    image_data[i] = static_cast<double>(data[i]) / 255.0; 
  
  stbi_image_free(data);
  return image_data;
}