#include "Neuron.h"
Neuron::Neuron(double* neuron_weights, double* neuron_bias, double* layer_input_features, int number_of_features) :
	neuron_weights(neuron_weights), neuron_bias(neuron_bias), layer_input_features(layer_input_features), number_of_features(number_of_features)
{
	derived_value = 0;
	activation_value = 0;
}

// compute activation value of neuron
double Neuron::reLU_activation_function()
{
	// activation_value will act us a running sum
	activation_value = 0;

	for (int f = 0; f < number_of_features; f++)
		activation_value += layer_input_features[f] * neuron_weights[f];

	activation_value += *neuron_bias;

	// relu function application
	if (activation_value <= 0)
		activation_value = 0;

	return activation_value;
}

// setter/mutator methods
void Neuron::set_derived_value(double d)
{ derived_value = d; }

double Neuron::get_derived_value()
{ return derived_value; }
