#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork(int neurons_each_layer[])
{
	// store the number of layers
	this->number_of_layers = sizeof(neurons_each_layer) / sizeof(int);

	// create a new array that will store pointers to layer objects of the inputted number of layers
	layers = new Layer* [number_of_layers];

	// the 0th/input layer will have 0 weights, but will have the same amount of neurons as features
	layers[0] = new InputLayer(neurons_each_layer[0]);

	// each nth layer will store the amount of neurons in the n-1th layer
	// this is so that we can make the number of corresponding number of weights for the nth layer, given that we
	// take the n-1th neurons as input as an activation vector
	for (int i = 1; i < number_of_layers; i++)
		layers[i] = new Layer(neurons_each_layer[i], neurons_each_layer[i - 1]);
}

void NeuralNetwork::train_network(int features[])
{

}

// predict based off an input of features
void NeuralNetwork::predict_network(int features[])
{
	
}