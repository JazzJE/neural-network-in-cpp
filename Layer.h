#pragma once

#include "Neuron.h"

class Layer
{
private:
	Neuron* neurons;
	int number_of_neurons_in_layer;
	int number_of_neurons_in_prior_layer;

public:

	// constructor to initialize neurons 
	Layer(int number_of_neurons, int number_of_neurons_in_prior_layer);

	// getter/accessor methods
	int get_number_of_neurons_in_layer() { return number_of_neurons_in_layer; }

	// setter/mutator methods
	void set_number_of_neurons_in_layer(int n) { number_of_neurons_in_layer = n; }

};

