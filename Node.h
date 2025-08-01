#ifndef NODE_H
#define NODE_H

#include <vector>
#include <functional>

class Node {
public:
    Node(int input_size = 0);
    Node(int input_size, std::function<double(double)> activation_function);

    void forward(const std::vector<double>& inputs);
    void update_weight(const std::vector<double> &predicted, const std::vector<double> & actual );
    void set_data(double new_data);
    void set_weights(const std::vector<double> &new_weights);
    double get_data();

private:
    double data;
    double bias;
    int input_size;

    std::vector<double> weights;
    std::function<double(double)> activation;
};

#endif 