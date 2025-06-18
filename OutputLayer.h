#pragma once
#include "DenseLayer.h"
#include "OutputNeuron.h"

// the output layer will functionally have the same attributes as the normal layer; the only difference will be in the methods, 
// which are optimized for just one (output) neuron
class OutputLayer : public DenseLayer
{
public:
	
	OutputLayer(double** layer_weights, double* layer_biases, double** layer_means_and_variances, double** layer_scales_and_shifts,
		double** training_layer_activation_arrays, double* layer_activation_array, int batch_size, int number_of_features,
		int number_of_neurons, double* regularization_rate, double* learning_rate);

	// delete the output neuron array
	~OutputLayer();

	// no for loops; just a single value
	void compute_activation_array();
	void training_compute_activation_arrays();
};

