#pragma once
#include "Neuron.h"
// functionally will have the same attributes as a regular neuron; HOWEVER, this neuron will only calculate using the linear
// transformation function with no other modifications to the activation
class OutputNeuron : public Neuron
{
public:

	OutputNeuron(double* neuron_weights, double* neuron_bias, double* mean_and_variance, double* scale_and_shift,
		double** training_input_features, double** training_activation_arrays, double* input_features, double* activation_array,
		int number_of_features, int batch_size, int neuron_number);

	// these methods will be different in that they will only use the linear transformation method
	void compute_activation_value() override;
	void training_compute_activation_values() override;

};

