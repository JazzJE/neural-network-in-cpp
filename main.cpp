#include <iostream>
#include "NeuralNetwork.h"
#include "Layer.h"
#include "Neuron.h"

// driver for class
int main()
{
	// turn the image into 

	// specify how many layers there should be in the network and how many neurons for each layer
	int number_of_layers = 3;
	int number_of_neurons_each_layer[] = { 5, 10, 1 };

	// the first argument should be the number of layers
	// the second argument should be an array for the number neurons in each layer
	// i.e., (3, [5, 10, 1]) = 3 layers — input layer = 5 features; 1st hidden layer = 10 neurons;output layer = 1 neuron;
	NeuralNetwork n(number_of_layers, number_of_neurons_each_layer);

	n.train_from_data();
}