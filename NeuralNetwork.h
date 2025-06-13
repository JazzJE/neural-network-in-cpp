#pragma once
#include "DenseLayer.h"
#include "MemoryFunctions.h"
#include "StatisticsFunctions.h"
#include <fstream>
#include <cstdlib>
#include <iostream>

class NeuralNetwork
{
private:

	const int network_number_of_features;
	const int batch_size;

	DenseLayer** const hidden_layers;
	const int* number_of_neurons_each_hidden_layer;
	const int number_of_hidden_layers;
	DenseLayer* const output_layer;

	double* const regularization_rate;
	double* const learning_rate;

	double*** const network_weights;
	double** const network_biases;
	double** const network_running_means_and_variances;
	double** const network_scales_and_shifts;

public:

	// initialize all the variables
	NeuralNetwork(double*** weights, double** biases, double** means_and_variances, double** scales_and_shifts, 
		const int* number_of_neurons_each_hidden_layer, int net_number_of_neurons, int number_of_hidden_layers, 
		int number_of_features, int batch_size, double learning_rate, double regularization_rate);

	// destructor to delete neural network
	~NeuralNetwork();

	// train the neural network five times based on the number of training samples
	void five_fold_train(double** training_features, double* target_values, int number_of_samples);
	void mini_batch_descent(double*** best_weights, double** best_biases, double** training_features_normalized, double* target_values,
		int lower_cross_validation_index, int higher_cross_validation_index, int number_of_samples);

	// calculate a value based on the current weights and biases as well as the input features
	double calculate_prediction(double* input_features);

	// mutator/setter methods for rates
	void set_regularization_rate(double r_rate);
	void set_learning_rate(double l_rate);

};

