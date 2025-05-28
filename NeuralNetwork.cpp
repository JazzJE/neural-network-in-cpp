#include "NeuralNetwork.h"

// initialize each hidden layer with their...
		// weights,
		// biases,
		// the number of weights they will have (which is the number of neurons in the previous layer but number of features for first layer),
		// and the number of neurons they will have

NeuralNetwork::NeuralNetwork(double*** weights, double** biases, const int* number_of_neurons_each_hidden_layer,
	int number_of_hidden_layers, int number_of_features, double learning_rate, double regularization_rate) : 
	
	network_number_of_features(number_of_features), number_of_neurons_each_hidden_layer(number_of_neurons_each_hidden_layer),
	number_of_hidden_layers(number_of_hidden_layers), regularization_rate(regularization_rate), learning_rate(learning_rate), 
	network_weights(weights), network_biases(biases), hidden_layers(new DenseLayer*[number_of_hidden_layers]), 
	
	// output layer
	output_layer(new DenseLayer(weights[number_of_hidden_layers], biases[number_of_hidden_layers],
		number_of_neurons_each_hidden_layer[number_of_hidden_layers - 1], 1))
{
	// first hidden layer
	hidden_layers[0] = new DenseLayer(weights[0], biases[0], number_of_features, number_of_neurons_each_hidden_layer[0]);

	// rest of the layers
	for (int l = 1; l < number_of_hidden_layers; l++)
		hidden_layers[l] = new DenseLayer(weights[l], biases[l], number_of_neurons_each_hidden_layer[l - 1], 
			number_of_neurons_each_hidden_layer[l]);
}

// delete all dynamically allocated objects
NeuralNetwork::~NeuralNetwork()
{

}

// train the neural network five times based on the number of training samples
void NeuralNetwork::five_fold_train(double** training_features, double* target_values, int number_of_training_samples)
{
	int lower_cross_validation_index, higher_cross_validation_index;
	int samples_per_division = number_of_training_samples / 5;
	double* means = new double[network_number_of_features];
	double* std_devs = new double[network_number_of_features];

	// for each training period of the neural network, use the lower 
	for (int i = 0; i < 4; i++)
	{
		lower_cross_validation_index = i * samples_per_division;
		higher_cross_validation_index = (i + 1) * samples_per_division - 1;
		mini_batch_descent(training_features, target_values, lower_cross_validation_index, higher_cross_validation_index,
			number_of_training_samples);
	}

	// use all the remaining training sets as the cross validation set
	lower_cross_validation_index = 4 * samples_per_division;
	higher_cross_validation_index = number_of_training_samples - 1;
	mini_batch_descent(training_features, target_values, lower_cross_validation_index, higher_cross_validation_index, 
		number_of_training_samples);

	delete[] means;
	delete[] std_devs;
}

// run mini-batch gradient descent on the provided fold
void NeuralNetwork::mini_batch_descent(double** training_features, double* target_values, int lower_validation_index,
	int higher_validation_index, int number_of_training_samples)
{
	// allocate memory to store the best current weights for the neural network
	double*** best_weights = allocate_memory_for_weights(number_of_neurons_each_hidden_layer, number_of_hidden_layers, 
		network_number_of_features);


}

// return a value based on the current weights and biases as well as the input features
double NeuralNetwork::calculate_prediction(double* input_features, double target_value)
{
	double* activation_array = input_features;

	// for the first layer
	// copy the previous layer's activation array into the next layer's input features
	for (int f = 0; f < network_number_of_features; f++)
		hidden_layers[0]->get_layer_input_features()[f] = activation_array[f];
	activation_array = hidden_layers[0]->compute_activation_array();

	// for every layer
	for (int l = 1; l < number_of_hidden_layers; l++)
	{
		// copy the previous layer's activation array into the next layer's input features
		for (int f = 0; f < hidden_layers[l]->get_number_of_features(); f++)
			hidden_layers[l]->get_layer_input_features()[f] = activation_array[f];

		// note that because we are copying the activation array into the next layer's input features, the activation_array must be deleted
		// after copying; this is because the activation array is unused after copying
		delete[] activation_array;

		activation_array = hidden_layers[l]->compute_activation_array();
	}

	// copy the output features of the last layer into the output layer
	for (int f = 0; f < output_layer->get_number_of_features(); f++)
		output_layer->get_layer_input_features()[f] = activation_array[f];
	delete[] activation_array;

	// output layer will only return a dynamic array of one value, which needs to be deleted before returning the output
	activation_array = output_layer->compute_activation_array();
	double output_value = *activation_array;
	delete[] activation_array;

	return output_value;
	
}

// mutator/setter methods for rates
void NeuralNetwork::set_regularization_rate(double r_rate)
{ regularization_rate = r_rate; }
void NeuralNetwork::set_learning_rate(double l_rate)
{ learning_rate = l_rate; }
