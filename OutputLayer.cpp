#include "OutputLayer.h"
OutputLayer::OutputLayer(double** layer_weights, double* layer_biases, double** layer_means_and_variances, double** layer_scales_and_shifts,
	double** training_layer_activation_arrays, double* layer_activation_array, int batch_size, int number_of_features,
	int number_of_neurons, double* regularization_rate, double* learning_rate) : DenseLayer(layer_weights, layer_biases, 
		layer_means_and_variances, layer_scales_and_shifts, training_layer_activation_arrays, layer_activation_array, 
		batch_size, number_of_features, number_of_neurons, regularization_rate, learning_rate)
{ 
	// instead of creating a neuron object within the neurons, create a singular neuron of the "OutputNeuron" class
	delete* neurons;
	*neurons = new OutputNeuron(layer_weights[0], &layer_biases[0], layer_means_and_variances[0], layer_scales_and_shifts[0],
		training_layer_input_features, training_layer_activation_arrays, layer_input_features, layer_activation_array,
		number_of_features, batch_size, 0);
}

// delete the output neuron array
OutputLayer::~OutputLayer()
{
	delete[] layer_activation_array;
	deallocate_memory_for_training_features(training_layer_activation_arrays, batch_size);
}

// for the singular output neuron
void OutputLayer::compute_activation_array()
{ (*neurons)->compute_activation_value(); }

// for the outputs of the training arrays
void OutputLayer::training_compute_activation_arrays()
{ (*neurons)->training_compute_activation_values(); }