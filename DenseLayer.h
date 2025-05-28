#pragma once
#include "Neuron.h"
class DenseLayer
{
private:

	Neuron** const neurons;
	int number_of_neurons;

	const int number_of_features;
	double* const layer_input_features;

public:

	// constructor that will initialize each neuron inside of the layer
	DenseLayer(double** layer_weights, double* layer_biases, int number_of_features, int number_of_neurons);

	// delete all the dynamically allocated objects
	~DenseLayer();

	// layer will go through each neuron and return a dynamically allocated array of all their values
	double* compute_activation_array();
	
	// getter/accessor methods
	double* get_layer_input_features();
	double get_number_of_features();

};

