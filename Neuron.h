#pragma once
class Neuron
{
private:

	double* const neuron_weights;
	double* const neuron_bias;

	double* const input_features;
	const int number_of_features;

	double derived_value;
	double activation_value;

public:

	// constructor
	Neuron(double* neuron_weights, double* neuron_bias, double* input_features, int number_of_features);

};

