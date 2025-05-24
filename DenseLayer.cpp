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
	input_features(new double[number_of_features])
{
	for (int n = 0; n < number_of_neurons; n++)
		neurons[n] = new Neuron(layer_weights[n], &layer_biases[n], input_features, number_of_features);
}