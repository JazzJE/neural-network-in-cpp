#pragma once

#include "Neuron.h"

class Layer
{
protected:
	Neuron** neurons;
	int number_of_neurons_in_layer;
	int number_of_neurons_in_prior_layer;
	double* input_features;

public:

	// constructor to initialize neurons 
	Layer(int number_of_neurons_in_layer, int number_of_neurons_in_prior_layer);

	// output an array of values 
	virtual double* output_activation_array();
};

