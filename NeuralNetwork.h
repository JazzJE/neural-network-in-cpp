#pragma once

#include "Layer.h"
#include "InputLayer.h"

class NeuralNetwork
{
private:
	Layer** layers;
	int number_of_layers;

public:

	// constructor to initialize the number of layers in the network and the number of neurons for each layer
	NeuralNetwork(int neurons_each_layer[]);

	// method to train the neural network off of a specific example's features
	void train_network(int features[]);

	// method to predict the output value of a specific example's features
	void predict_network(int features[]);

};

