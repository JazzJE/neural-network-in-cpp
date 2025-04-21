#pragma once
class Neuron
{
private:
	double* weights;
	int number_of_weights;
	double activation_value;
	double derived_value;
	double* input_features;

public:
	
	// constructor to initialize weights and the like
	Neuron(int number_of_weights);
	
	// train neuron
	void train_neuron();

	// compute activation the activation value of this neuron
	double compute_activation_value();
};

