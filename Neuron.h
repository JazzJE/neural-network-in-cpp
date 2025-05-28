#pragma once
class Neuron
{
private:

	double* const neuron_weights;
	double* const neuron_bias;

	double* const layer_input_features;
	const int number_of_features;

	double derived_value;
	double activation_value;

public:

	// constructor
	Neuron(double* neuron_weights, double* neuron_bias, double* layer_input_features, int number_of_features);

	// function to compute the activation value and derived value
	double reLU_activation_function();

	// mutator/setter methods
	void set_derived_value(double d);

	// acessor/getter methods
	double get_derived_value();
};

