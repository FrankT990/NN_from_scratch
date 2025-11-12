#include <vector>
#include <string>
#include <utility>

// folder_path: path to train / test folder
// classes: list of class folder names
// input_size: input length of each sample

// returns vector of <input_vector, one_hot_label> pairs
std::vector<std::pair<std::vector<double>, std::vector<double> > > 
load_dataset(const std::string &folder_path, 
             const std::vector<std::string>& classes,
             int &input_size);

std::vector<double> load_single_image(const std::string& image_path, int size);