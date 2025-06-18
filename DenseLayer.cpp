#include "DenseLayer.h"

// each neuron should store...
	// the weights of the neuron
	// the memory address of the bias value of the neuron
	// the number of weights/features
	// a pointer to the input features of the layer, such that when the layer's input features is updated, all the neurons are effectively
		// updated as well
DenseLayer::DenseLayer(double** layer_weights, double* layer_biases, double** layer_means_and_variances, double** layer_scales_and_shifts,
	double** training_layer_activation_values, double* layer_activation_array, int batch_size, int number_of_features, int number_of_neurons,
	double* regularization_rate, double* learning_rate)
	: number_of_features(number_of_features), number_of_neurons(number_of_neurons), batch_size(batch_size), 
	neurons(new Neuron*[number_of_neurons]),
	
	// assign the activation arrays/output arrays that we are outputting to as the input arrays/feature arrays of the n + 1th layer
	training_layer_activation_arrays(training_layer_activation_values),
	layer_activation_array(layer_activation_array),

	// create new inputs that can then be used for the n - 1th layer
	training_layer_input_features(allocate_memory_for_training_features(batch_size, number_of_features)),
	layer_input_features(new double[number_of_features])
{
	for (int n = 0; n < number_of_neurons; n++)
		neurons[n] = new Neuron(layer_weights[n], &layer_biases[n], layer_means_and_variances[n], layer_scales_and_shifts[n],
			training_layer_input_features, training_layer_activation_arrays, layer_input_features, layer_activation_array,
			number_of_features, batch_size, n);
}

// deallocate the neurons and the input features; the nth layer will have its output layer activation arrays be deallocated by the 
// (n + 1)th layer as they refer to the same thing
DenseLayer::~DenseLayer()
{
	for (int n = 0; n < number_of_neurons; n++)
		delete neurons[n];
	delete[] neurons;

	delete[] layer_input_features;
	deallocate_memory_for_training_features(training_layer_input_features, batch_size);
}

// calculate each neuron's activation values
void DenseLayer::compute_activation_array()
{
	for (int n = 0; n < number_of_neurons; n++)
		neurons[n]->compute_activation_value();
}

// calculate each sample's activation_values
void DenseLayer::training_compute_activation_arrays()
{
	for (int n = 0; n < batch_size; n++)
		neurons[n]->training_compute_activation_values();
}

// return where the layer inputs will be stored
double* DenseLayer::get_layer_input_features() const
{ return layer_input_features; }
// return the activation array
double* DenseLayer::get_layer_activation_array() const
{ return layer_activation_array; }

// return where the training layer input features will be stored
double** DenseLayer::get_training_layer_input_features() const
{ return training_layer_input_features; }
// return the training activation array
double** DenseLayer::get_training_layer_activation_arrays() const
{ return training_layer_activation_arrays; }