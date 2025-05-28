#include "MemoryAllocFunctions.h"

// methods to allocate memory for weights

	// allocate 3d array for weights via 3d pointer
		// the 3d pointer can switch to any of the 2d layer pointers
		// each 2d layer pointer will store an array of 1d pointers that each point to the weights of a given neuron
		// each 1d pointer will be pointers to arrays of weights of a given neuron
double*** allocate_memory_for_weights(const int* number_of_neurons_each_hidden_layer, int number_of_hidden_layers, int number_of_features)
{
	double*** weights = new double** [number_of_hidden_layers + 1];

	// allocate memory for the first layer using the number of features
	// add one to also store value of the bias value along with number of features
	weights[0] = new double* [number_of_neurons_each_hidden_layer[0]];
	for (int n = 0; n < number_of_neurons_each_hidden_layer[0]; n++)
		weights[0][n] = new double[number_of_features];

	// allocate memory for each subsequent layer
	for (int l = 1; l < number_of_hidden_layers; l++)
	{
		weights[l] = new double* [number_of_neurons_each_hidden_layer[l]];

		// number of features of given layer l is the number of neurons in the previous layer (l - 1)
		// add one to also have a place to store bias
		for (int n = 0; n < number_of_neurons_each_hidden_layer[l]; n++)
			weights[l][n] = new double[number_of_neurons_each_hidden_layer[l - 1]];
	}

	// allocate memory for output layer with only one neuron
	// the number_of_hidden_layers is equal to the index of the last/output layer pointer
	weights[number_of_hidden_layers] = new double*;
	weights[number_of_hidden_layers][0] = new double[number_of_neurons_each_hidden_layer[number_of_hidden_layers - 1]];

	return weights;
}

// allocate 2d array for biases via 2d pointer
	// the 2d pointer can point to any of the 1d layer pointers
	// each 1d layer pointer will point to the beginning of an array of doubles where each double
		// represents each neuron's bias value inside of the layer
double** allocate_memory_for_biases(const int* number_of_neurons_each_hidden_layer, int number_of_hidden_layers, int number_of_features)
{
	double** biases = new double* [number_of_hidden_layers + 1];

	// allocate memory for bias values of the first layer
	biases[0] = new double[number_of_neurons_each_hidden_layer[0]];

	// allocate memory for each subsequent layer
	for (int l = 1; l < number_of_hidden_layers; l++)
		biases[l] = new double[number_of_neurons_each_hidden_layer[l]];

	// allocate memory for last layer/output layer with one neuron
	biases[number_of_hidden_layers] = new double;

	return biases;
}

// allocate 2d array for training samples via 2d pointer
	// the 2d pointer can point to any 1d sample pointers
	// each 1d sample pointer will store the features of the ith example
double** allocate_memory_for_training_samples(int number_of_samples, int number_of_features)
{
	double** training_samples = new double* [number_of_samples];
	for (int i = 0; i < number_of_samples; i++)
		training_samples[i] = new double[number_of_features];

	return training_samples;
}

// allocate 1d array for training samples via 1d pointer
	// each will store the respective actual ith value of the ith samples features
double* allocate_memory_for_target_values(int number_of_samples)
{
	double* target_values = new double[number_of_samples];

	return target_values;
}