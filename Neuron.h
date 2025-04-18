#pragma once
class Neuron
{
private:
	double* weights;
	double activation_value;
	double derived_value;
	double* prior_layer_activations;

public:
	
	// constructor to initialize weights and the like
	Neuron(int number_of_weights);
	
	// getter/accessor methods
	double get_activation_value();
	double get_derived_value();

	
};

