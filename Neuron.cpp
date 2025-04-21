#include "Neuron.h"
#include <cmath>
#include<random>

Neuron::Neuron(int number_of_weights)
{	
	// store the number of weights inside this neuron
	this->number_of_weights = number_of_weights;

	// generate a random seed for random numbers
	std::random_device rd;
	std::mt19937 gen(rd());

	// the number of weights is equal to the number of neurons which come from the n-1th layer
	double stddev = sqrt(2.0 / number_of_weights);  // He initialization formula
	std::normal_distribution<double> dist(0.0, stddev);

	// the number of weights in a given neuron within the layer will be equal to the number of neurons in the previous layer
	// this is because each of the previous layer's neurons will output a value
	weights = new double[number_of_weights];

	// assign the weights of the neuron to random values using He initialization
	for (int i = 0; i < number_of_weights; i++)
		weights[i] = dist(gen);
}

// method for a given neuron to return its activation value using a relu function
double Neuron::compute_activation_value()
{
	double activation_value = 0;

	for (int i = 0; i < number_of_weights; i++)
		activation_value += weights[i] * input_features[i];

	// relu function implementation
	if (activation_value < 0) return 0;
	else return activation_value;
}