#pragma once

#include "Layer.h"

class NeuralNetwork
{
private:
	Layer** layers;
	int number_of_layers;
	int net_number_of_neurons;

public:

	// constructor to initialize the number of layers in the network and the number of neurons for each layer
	NeuralNetwork(int number_of_layers, int* neurons_each_layer);

	// method to actually train neural network to be more accurate
	void train_from_data();
};

