#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork(int number_of_layers, int* neurons_each_layer )
{
	// store how many layers there will be in the network
	this->number_of_layers = number_of_layers;

	// create a new array that will store pointers to layer objects of the inputted number of layers
	layers = new Layer* [number_of_layers];

	net_number_of_neurons = 0;

	// each nth layer will store the amount of neurons in the n-1th layer
	// this is so that we can make the number of corresponding number of weights for the nth layer, given that we
	// take the n-1th neurons as input as an activation vector
	for (int i = number_of_layers; i > 0; i--)
	{
		layers[i] = new Layer(neurons_each_layer[i], neurons_each_layer[i - 1]);
		net_number_of_neurons += neurons_each_layer[i];
	}

	// the 0th/input layer will have 0 neurons in the previous layer
	layers[0] = new Layer(neurons_each_layer[0], 0);
}