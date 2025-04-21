#include <iostream>
#include <fstream>
#include "NeuralNetwork.h"
#include "Layer.h"
#include "Neuron.h"

// data set used: https://www.kaggle.com/datasets/meharshanali/amazon-stocks-2025
int number_of_samples = 6988;
int number_of_fields = 6;

// driver for class
int main()
{
	// specify the number of neurons you want to have in the layers
	// i.e., {5, 10, 1} = 3 layers — input layer = 5 features; 1st hidden layer = 10 neurons;output layer = 1 neuron;
	// WARNING: ensure that the first number is equal to the number of features of the samples; this is our 0th/input layer
	int number_of_neurons_each_layer[] = { number_of_fields, 10, 1 };
	NeuralNetwork n(number_of_neurons_each_layer);
}