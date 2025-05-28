#include "DenseLayer.h"

// each neuron should store...
	// the weights of the neuron
	// the memory address of the bias value of the neuron
	// the number of weights/features
	// a pointer to the input features of the layer, such that when the layer's input features is updated, all the neurons are effectively
		// updated as well
DenseLayer::DenseLayer(double** layer_weights, double* layer_biases, int number_of_features, int number_of_neurons)
	: number_of_features(number_of_features), number_of_neurons(number_of_neurons), neurons(new Neuron*[number_of_neurons]),
	
	// this array will store the output values of the previous layer's activations by copying the activations into this array
	// this allows the features to update across all neurons simultaneously rather than one at a time
	layer_input_features(new double[number_of_features])
{
	for (int n = 0; n < number_of_neurons; n++)
		neurons[n] = new Neuron(layer_weights[n], &layer_biases[n], layer_input_features, number_of_features);
}

// collect all the activations in a single dynamically allocated array
double* DenseLayer::compute_activation_array()
{
	double* activation_array = new double[number_of_neurons];

	for (int n = 0; n < number_of_neurons; n++)
		activation_array[n] = neurons[n]->reLU_activation_function();

	return activation_array;
}

// return where the layer inputs will be stored
double* DenseLayer::get_layer_input_features()
{  return layer_input_features;  }

// return the number of features the layer will receive
double DenseLayer::get_number_of_features()
{  return number_of_features;  }