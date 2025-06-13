#include "Neuron.h"
Neuron::Neuron(double* neuron_weights, double* neuron_bias, double* mean_and_variance, double* scale_and_shift,
	double** training_input_features, double** training_activation_arrays, double* input_features, double* activation_array,
	int number_of_features, int neuron_number) :
	neuron_weights(neuron_weights), neuron_bias(neuron_bias), running_mean(&mean_and_variance[0]), running_variance(&mean_and_variance[1]),
	scale(&scale_and_shift[0]), shift(&scale_and_shift[1]), number_of_features(number_of_features), momentum(0.9), 
	neuron_number(neuron_number),
	
	// link to the n-1th layer and the n+1th layer
	input_features(input_features), activation_array(activation_array),
	training_input_features(training_input_features), training_activation_arrays(training_activation_arrays)

{
	derived_value = 0;
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
