#pragma once
#include "DenseLayer.h"
#include <fstream>
class NeuralNetwork
{
private:

	const int number_of_features;

	DenseLayer** const hidden_layers;
	const int* number_of_neurons_each_hidden_layer;
	const int number_of_hidden_layers;
	DenseLayer* const output_layer;

	double regularization_rate;
	double learning_rate;

	const double*** const network_weights;
	const double** const network_biases;

public:

	// initialize all the variables
	NeuralNetwork(int number_of_features, int* number_of_neurons_each_hidden_layer, int number_of_hidden_layers, 
		double regularization_rate, double learning_rate, double*** network_weights, double** network_biases);

	// destructor to delete neural network
	~NeuralNetwork();

	// train the neural network five times based on the number of training samples
	void five_fold_train(double** training_features, double* target_values, int number_of_training_samples);
	void train(double** training_features, double* target_values, int initial_index, int final_index, int number_of_training_samples);

	// calculate a value based on the current weights and biases as well as the input features
	double calculate_prediction(double** input_features);

	// accessor/getter methods
	const double*** get_network_weights();
	const double** get_network_biases();

	// mutator/setter methods
	void set_regularization_rate(double r_rate);
	void set_learning_rate(double l_rate);

};

