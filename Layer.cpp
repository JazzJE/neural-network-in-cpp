#include "Layer.h"

// constructor to initialize neurons 
Layer::Layer(int number_of_neurons_in_layer, int number_of_neurons_in_prior_layer)
{
	// initialize private variables
	this->number_of_neurons_in_layer = number_of_neurons_in_layer;
	this->number_of_neurons_in_prior_layer = number_of_neurons_in_prior_layer;
	this->input_features = nullptr;

	// if there are no neurons in the previous layer, this means that the input layer we are initializing is the input layer
	// thus, do not unnecessarily create dynamically allocated neurons 
	// all the input layer will do is take in the array of input features and output them as the "activation array" for
	// the next layer
	if (number_of_neurons_in_prior_layer == 0) return;

	// else, dynamically allocate neurons to the layer
	else
	{
		// dynamically allocate pointers to dynamically allocated neuron objects
		neurons = new Neuron * [number_of_neurons_in_layer];

		// assign objects to each dynamically allocated pointer
		for (int i = 0; i < number_of_neurons_in_layer; i++)
			neurons[i] = new Neuron(number_of_neurons_in_prior_layer);
	}
}

// output the activation array of the current layer using the input layer
double* Layer::output_activation_array()
{
	double* activation_array = new double[number_of_neurons_in_layer];

	for (int i = 0; i < number_of_neurons_in_layer; i++)
		activation_array[i] = (neurons[i])->compute_activation_value();

	return activation_array;
}