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
void NeuralNetwork::five_fold_train(double** training_features, double* target_values, int number_of_samples)
{
	// create these pointers to store the best weights and best bias values for the current iteration
	double*** best_weights = allocate_memory_for_weights(number_of_neurons_each_hidden_layer, number_of_hidden_layers,
		network_number_of_features);
	double** best_biases = allocate_memory_for_biases(number_of_neurons_each_hidden_layer, number_of_hidden_layers);

	int lower_cross_validation_index, higher_cross_validation_index;
	int samples_per_fold = number_of_samples / 5;

	// for each training period of the neural network, use the lower 
	for (int i = 0; i < 4; i++)
	{
		lower_cross_validation_index = i * samples_per_fold;
		higher_cross_validation_index = (i + 1) * samples_per_fold - 1;

		double* training_means = calculate_features_means(training_features, network_number_of_features, number_of_samples,
			lower_cross_validation_index, higher_cross_validation_index);
		double* training_stddevs = calculate_features_stddevs(training_features, training_means,
			network_number_of_features, number_of_samples, lower_cross_validation_index, higher_cross_validation_index);
		
		double** training_features_normalized = calculate_normalized_features(training_features, number_of_samples, network_number_of_features, 
			training_means, training_stddevs);

		mini_batch_descent(best_weights, best_biases, training_features_normalized, target_values, 
			lower_cross_validation_index, higher_cross_validation_index, number_of_samples);

		delete[] training_means;
		delete[] training_stddevs;
		deallocate_memory_for_training_features(training_features_normalized, number_of_samples);
	}

	// use all the remaining training sets as the cross validation set
	lower_cross_validation_index = 4 * samples_per_fold;
	higher_cross_validation_index = number_of_samples - 1;

	double* training_means = calculate_features_means(training_features, network_number_of_features, number_of_samples, 
		lower_cross_validation_index, higher_cross_validation_index);
	double* training_stddevs = calculate_features_stddevs(training_features, training_means,
		network_number_of_features, number_of_samples, lower_cross_validation_index, higher_cross_validation_index);
	
	double** training_features_normalized = calculate_normalized_features(training_features, number_of_samples, network_number_of_features, 
		training_means, training_stddevs);

	mini_batch_descent(best_weights, best_biases, training_features_normalized, target_values,
		lower_cross_validation_index, higher_cross_validation_index, number_of_samples);

	// deallocate all memory
	delete[] training_means;
	delete[] training_stddevs;
	deallocate_memory_for_training_features(training_features_normalized, number_of_samples);
	deallocate_memory_for_weights(best_weights, number_of_neurons_each_hidden_layer, number_of_hidden_layers);
	deallocate_memory_for_biases(best_biases, number_of_hidden_layers);
}

// run mini-batch gradient descent on the provided fold
void NeuralNetwork::mini_batch_descent(double*** best_weights, double** best_biases, double** training_features_normalized, double* target_values,
	int lower_validation_index, int higher_validation_index, int number_of_samples)
{
	const int batch_size = 64;
	const int patience = 5;
	double best_mse = -1, current_mse = 0;
	int selected_sample_indices[batch_size];

	// count the number of times the network has failed to product a smaller mse value, 
	// and end training with this fold when it meets the patience value
	int failed_epochs = 0;

	while (failed_epochs < patience)
	{
		current_mse = 0;	
		// select random sample indices for the batch
		for (int i = 0; i < batch_size; i++)
		{
			// select a random index
			selected_sample_indices[i] = std::rand() % number_of_samples;

			// while the random index is inside the range of the cross-validation set, select a new index
			while (selected_sample_indices[i] >= lower_validation_index && selected_sample_indices[i] <= higher_validation_index)
				selected_sample_indices[i] = std::rand() % number_of_samples;
		}

		for (int i = 0; i < batch_size; i++)
			current_mse += pow(calculate_prediction(training_features_normalized[selected_sample_indices[i]]) - 
				target_values[selected_sample_indices[i]], 2.0);
		current_mse /= (2 * batch_size);

		std::cout << "\n\tThe current mean_squared error is " << current_mse
			<< "\n\tThe previous mean_squared error is " << previous_mse << std::endl;

		if (current_mse > previous_mse)
			failed_epochs++;
		else
		{
			
		}
		std::cout << "\nThe number of failed epochs is " << failed_epochs << std::endl;
	}
}

// return a value based on the current weights and biases as well as the input features
double NeuralNetwork::calculate_prediction(double* input_features)
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
