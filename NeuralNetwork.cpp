#include "NeuralNetwork.h"

// initialize each hidden layer with their...
		// weights,
		// biases,
		// scales and shifts,
		// running means and variances,
		// the number of weights they will have (which is the number of neurons in the previous layer but number of features for first layer),
		// and the number of neurons they will have

NeuralNetwork::NeuralNetwork(double*** weights, double** biases, double** means_and_variances, double** scales_and_shifts,
	const int* number_of_neurons_each_hidden_layer, int net_number_of_neurons, int number_of_hidden_layers, int number_of_features,
	int batch_size, double learning_rate, double regularization_rate) :
	
	network_number_of_features(number_of_features), number_of_neurons_each_hidden_layer(number_of_neurons_each_hidden_layer),
	number_of_hidden_layers(number_of_hidden_layers), regularization_rate(new double(regularization_rate)), learning_rate(new double(learning_rate)), 
	network_weights(weights), network_biases(biases), network_running_means_and_variances(means_and_variances), 
	network_scales_and_shifts(scales_and_shifts), batch_size(batch_size), hidden_layers(new DenseLayer*[number_of_hidden_layers]), 
	
	// output layer
	output_layer(new DenseLayer(weights[number_of_hidden_layers], biases[number_of_hidden_layers],
		&network_running_means_and_variances[net_number_of_neurons - 1], &network_scales_and_shifts[net_number_of_neurons - 1],

		// the output for the activation array of training and actual predictions will be "one" features for any sample
		allocate_memory_for_training_features(batch_size, 1), new double[1], 
		batch_size, number_of_neurons_each_hidden_layer[number_of_hidden_layers - 1], 1, this->regularization_rate, this->learning_rate))

{
	*(this->regularization_rate) = regularization_rate;
	*(this->learning_rate) = learning_rate;

	// "hooking" refers to connecting any given nth layer's output features to the (n + 1)th layer's input features via pointers

	// if there is only hidden layer, then hook the output and input layer to this layer
	if (number_of_hidden_layers == 1)
		hidden_layers[0] = new DenseLayer(weights[0], biases[0], &network_running_means_and_variances[0],
			&network_scales_and_shifts[0], output_layer->get_training_layer_input_features(), output_layer->get_layer_input_features(), 
			batch_size,	network_number_of_features, number_of_neurons_each_hidden_layer[0], this->regularization_rate, this->learning_rate);
	
	// else, if there are n layers
	else
	{
		// this will refer to the index of the means and variances & scales and shifts that the layer is allotted to for its neurons
		int current_index = net_number_of_neurons - 1;

		// hook the nth layer to the output layer
		current_index -= number_of_neurons_each_hidden_layer[number_of_hidden_layers - 1];
		hidden_layers[number_of_hidden_layers - 1] = new DenseLayer(weights[number_of_hidden_layers - 1], biases[number_of_hidden_layers - 1], 
			&network_running_means_and_variances[0], &network_scales_and_shifts[0], output_layer->get_training_layer_input_features(), 
			output_layer->get_layer_input_features(), batch_size, number_of_neurons_each_hidden_layer[number_of_hidden_layers - 2], 
			number_of_neurons_each_hidden_layer[number_of_hidden_layers - 1], this->regularization_rate, this->learning_rate);

		// for each current layer
		for (int l = number_of_hidden_layers - 1; l > 1; l--)
		{
			current_index -= number_of_neurons_each_hidden_layer[l - 1];
			hidden_layers[l - 1] = new DenseLayer(weights[l - 1], biases[l - 1], &network_running_means_and_variances[current_index], 
				&network_scales_and_shifts[current_index], hidden_layers[l]->get_training_layer_input_features(), 
				hidden_layers[l]->get_layer_input_features(), batch_size, number_of_neurons_each_hidden_layer[l - 2], 
				number_of_neurons_each_hidden_layer[l - 1], this->regularization_rate, this->learning_rate);
		}

		// hook the 1st layer to the input features of the 2nd layer, but also hook to the input features of the input layer
		// note that the current_index will always equal to 0 at this point
		hidden_layers[0] = new DenseLayer(weights[0], biases[0], &network_running_means_and_variances[0], &network_scales_and_shifts[0], 
			hidden_layers[1]->get_training_layer_input_features(), hidden_layers[1]->get_layer_input_features(), batch_size, 
			network_number_of_features,	number_of_neurons_each_hidden_layer[0], this->regularization_rate, this->learning_rate);
	}

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
			<< "\n\tThe previous mean_squared error is " << best_mse << std::endl;

		if (current_mse > best_mse)
			failed_epochs++;
		else
		{
			
		}
		std::cout << "\nThe number of failed epochs is " << failed_epochs << std::endl;
	}
}

// return a value based on the current weights and biases as well as the input features
double NeuralNetwork::calculate_prediction(double* normalized_input_features)
{
	// copy the normalized input features into the first layer's input array
	for (int f = 0; f < network_number_of_features; f++)
		hidden_layers[0]->get_layer_input_features()[f] = normalized_input_features[f];

	// for each layer
	
		// have compute the activation values

	// output layer will calculate a singular value and return that value as the result
	
}

// mutator/setter methods for rates
void NeuralNetwork::set_regularization_rate(double r_rate)
{ *regularization_rate = r_rate; }
void NeuralNetwork::set_learning_rate(double l_rate)
{ *learning_rate = l_rate; }
