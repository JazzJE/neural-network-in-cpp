#include "OutputNeuron.h"
OutputNeuron::OutputNeuron(double* neuron_weights, double* neuron_bias, double* mean_and_variance, double* scale_and_shift,
	double** training_input_features, double** training_activation_arrays, double* input_features, double* activation_array,
	int number_of_features, int batch_size, int neuron_number) : Neuron(neuron_weights, neuron_bias, mean_and_variance, scale_and_shift,
		training_input_features, training_activation_arrays, input_features, activation_array,
		number_of_features, batch_size, neuron_number)
{ }

// only do linear transformation to compute the activation value
void OutputNeuron::compute_activation_value()
{ linear_transform(); }

// only do linear transformations to compute each training activation value
void OutputNeuron::training_compute_activation_values()
{ training_linear_transform(); }