#include "NeuralNetwork.h"

// initialize each hidden layer with their...
		// weights,
		// biases,
		// the number of weights they will have (which is the number of neurons in the previous layer but number of features for first layer),
		// and the number of neurons they will have

NeuralNetwork::NeuralNetwork(int number_of_features, int* number_of_neurons_each_hidden_layer, int number_of_hidden_layers, 
	double regularization_rate, double learning_rate, double*** weights, double** biases) : number_of_features(number_of_features), number_of_neurons_each_hidden_layer(number_of_neurons_each_hidden_layer), 
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

}

void NeuralNetwork::train(double** training_features, double* target_values, int initial_index, int final_index, int number_of_training_samples)
{

}

// return a value based on the current weights and biases as well as the input features
double NeuralNetwork::calculate_prediction(double** input_features)
{

}

// accessor/getter methods
const double*** NeuralNetwork::get_network_weights()
{ return network_weights; }
const double** NeuralNetwork::get_network_biases()
{ return network_biases; }

// mutator/setter methods
void NeuralNetwork::set_regularization_rate(double r_rate)
{ regularization_rate = r_rate; }
void NeuralNetwork::set_learning_rate(double l_rate)
{ learning_rate = l_rate; }
