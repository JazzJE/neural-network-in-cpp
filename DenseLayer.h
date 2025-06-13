#pragma once
#include "Neuron.h"
#include "MemoryFunctions.h"
class DenseLayer
{
private:

	Neuron** const neurons;
	const int number_of_neurons;

	// for normal prediction and computation
	const int number_of_features;
	double* const layer_input_features;
	double* const layer_activation_array;

	const int batch_size;
	double** const training_layer_input_features;
	double** const training_layer_activation_arrays;

public:

	// constructor that will initialize each neuron inside of the layer
	DenseLayer(double** layer_weights, double* layer_biases, double** layer_means_and_variances, double** layer_scales_and_shifts,
		double** training_layer_activation_arrays, double* layer_activation_array, int batch_size, int number_of_features, 
		int number_of_neurons, double* regularization_rate, double* learning_rate);

	// delete all the dynamically allocated objects
	~DenseLayer();

	// layer will go through each neuron and return a dynamically allocated array of all their values
	double* compute_activation_array();
	
	// getter/accessor methods
	double* get_layer_input_features();
	double get_number_of_features();
	double** get_training_layer_input_features();

};

