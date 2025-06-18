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
	number_of_hidden_layers(number_of_hidden_layers), network_regularization_rate(new double(regularization_rate)), network_learning_rate(new double(learning_rate)), 
	network_weights(weights), network_biases(biases), network_means_and_variances(means_and_variances), 
	network_scales_and_shifts(scales_and_shifts), batch_size(batch_size), hidden_layers(new DenseLayer*[number_of_hidden_layers]), 
	net_number_of_neurons(net_number_of_neurons),
	
	// output layer
	output_layer(new OutputLayer(weights[number_of_hidden_layers], biases[number_of_hidden_layers],
		&means_and_variances[net_number_of_neurons - 1], &scales_and_shifts[net_number_of_neurons - 1],

		// the output for the activation array of training and actual predictions will be "one" features for any sample
		allocate_memory_for_training_features(batch_size, 1), new double[1], 
		batch_size, number_of_neurons_each_hidden_layer[number_of_hidden_layers - 1], 1, network_regularization_rate, network_learning_rate))

{
	*network_regularization_rate = regularization_rate;
	*network_learning_rate = learning_rate;

	// "hooking" refers to connecting any given nth layer's input features as the (n - 1)th layer's output activation values via pointers

	// if there is only hidden layer, then hook the output and input layer to this layer
	if (number_of_hidden_layers == 1)
		hidden_layers[0] = new DenseLayer(weights[0], biases[0], &means_and_variances[0],
			&scales_and_shifts[0], output_layer->get_training_layer_input_features(), output_layer->get_layer_input_features(), 
			batch_size,	number_of_features, number_of_neurons_each_hidden_layer[0], network_regularization_rate, network_learning_rate);
	
	// else, if there are n layers
	else
	{
		// this will refer to the index of the means and variances & scales and shifts that the layer is allotted to for its neurons
		int current_index = net_number_of_neurons - 1;

		// hook the nth layer to the output layer
		current_index -= number_of_neurons_each_hidden_layer[number_of_hidden_layers - 1];
		hidden_layers[number_of_hidden_layers - 1] = new DenseLayer(weights[number_of_hidden_layers - 1], biases[number_of_hidden_layers - 1], 
			&means_and_variances[0], &scales_and_shifts[0], output_layer->get_training_layer_input_features(), 
			output_layer->get_layer_input_features(), batch_size, number_of_neurons_each_hidden_layer[number_of_hidden_layers - 2], 
			number_of_neurons_each_hidden_layer[number_of_hidden_layers - 1], network_regularization_rate, network_learning_rate);

		// for each current layer
		for (int l = number_of_hidden_layers - 1; l > 1; l--)
		{
			current_index -= number_of_neurons_each_hidden_layer[l - 1];
			hidden_layers[l - 1] = new DenseLayer(weights[l - 1], biases[l - 1], &means_and_variances[current_index], 
				&scales_and_shifts[current_index], hidden_layers[l]->get_training_layer_input_features(), 
				hidden_layers[l]->get_layer_input_features(), batch_size, number_of_neurons_each_hidden_layer[l - 2], 
				number_of_neurons_each_hidden_layer[l - 1], network_regularization_rate, network_learning_rate);
		}

		// hook the 1st layer to the input features of the 2nd layer, but also hook to the input features of the input layer
		// note that the current_index will always equal to 0 at this point
		hidden_layers[0] = new DenseLayer(weights[0], biases[0], &means_and_variances[0], &scales_and_shifts[0], 
			hidden_layers[1]->get_training_layer_input_features(), hidden_layers[1]->get_layer_input_features(), batch_size, 
			network_number_of_features,	number_of_neurons_each_hidden_layer[0], network_regularization_rate, network_learning_rate);
	}

}

// delete all dynamically allocated objects
NeuralNetwork::~NeuralNetwork()
{
	for (int l = 0; l < number_of_hidden_layers; l++)
		delete hidden_layers[l];
	delete[] hidden_layers;

	delete output_layer;

	delete network_regularization_rate;
	delete network_learning_rate;

	deallocate_memory_for_weights(network_weights, number_of_neurons_each_hidden_layer, number_of_hidden_layers);
	deallocate_memory_for_biases(network_biases, number_of_hidden_layers);

	deallocate_memory_for_mv_or_ss(network_means_and_variances, net_number_of_neurons);
	deallocate_memory_for_mv_or_ss(network_scales_and_shifts, net_number_of_neurons);
}

// train the neural network five times based on the number of training samples
void NeuralNetwork::five_fold_train(double** training_features, double* target_values, int number_of_samples)
{
	// create these pointers to store the best weights and best bias values for the current iteration
	BestStateLoader bs_loader(network_weights, network_biases, network_means_and_variances, network_scales_and_shifts,
		number_of_neurons_each_hidden_layer, number_of_hidden_layers, net_number_of_neurons, network_number_of_features);

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

		mini_batch_descent(bs_loader, training_features_normalized, 
			target_values, lower_cross_validation_index, higher_cross_validation_index, number_of_samples);

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

	mini_batch_descent(bs_loader, training_features_normalized,
		target_values, lower_cross_validation_index, higher_cross_validation_index, number_of_samples);

	// deallocate all memory
	delete[] training_means;
	delete[] training_stddevs;
	deallocate_memory_for_training_features(training_features_normalized, number_of_samples);
}

// run mini-batch gradient descent on the provided fold
void NeuralNetwork::mini_batch_descent(BestStateLoader& bs_loader, double** training_features_normalized, double* target_values, 
	int lower_validation_index, int higher_validation_index, int number_of_samples)
{
	const int patience = 5;
	double best_mse, current_mse;
	int* selected_sample_indices = new int[batch_size];

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

	delete[] selected_sample_indices;
}

// return a value based on the current weights and biases as well as the input features
double NeuralNetwork::calculate_prediction(double* normalized_input_features)
{
	// copy the normalized input features into the first layer's input array
	for (int f = 0; f < network_number_of_features; f++)
		hidden_layers[0]->get_layer_input_features()[f] = normalized_input_features[f];

	// for each layer, have each compute their activation arrays
	for (int l = 0; l < number_of_hidden_layers; l++)
		hidden_layers[l]->compute_activation_array();

	// output layer will calculate a singular value and return that value as the result
	output_layer->compute_activation_array();
	return *(output_layer->get_layer_activation_array());
}

// calculate training predictions and return a dynamically allocated array of values to it
double* NeuralNetwork::calculate_training_predictions(double** normalized_input_features)
{
	// copy the normalized input features into the first layer's training input arrays
	for (int s = 0; s < batch_size; s++)
		for (int f = 0; f < network_number_of_features; f++)
			hidden_layers[0]->get_training_layer_input_features()[s][f] = normalized_input_features[s][f];

	// for each layer, compute their training activation arrays
	for (int l = 0; l < number_of_hidden_layers; l++)
		hidden_layers[l]->training_compute_activation_arrays();

	// output layer will calculate the batch size samples 
	output_layer->training_compute_activation_arrays();

	// copy the outputs into a dynamically allocated array to return from the function
	double* training_predictions = new double[batch_size];
}

// mutator/setter methods for rates
void NeuralNetwork::set_regularization_rate(double r_rate)
{ *network_regularization_rate = r_rate; }
void NeuralNetwork::set_learning_rate(double l_rate)
{ *network_learning_rate = l_rate; }



// initialize with the network's parameters
NeuralNetwork::BestStateLoader::BestStateLoader(double*** network_weights, double** network_biases, double** network_means_and_variances,
	double** network_scales_and_shifts, const int* number_of_neurons_each_hidden_layers, int number_of_hidden_layers,
	int net_number_of_neurons, int network_number_of_features) :
	current_weights(network_weights), current_biases(network_biases), current_means_and_variances(network_means_and_variances),
	current_scales_and_shifts(network_scales_and_shifts), number_of_neurons_each_hidden_layer(number_of_neurons_each_hidden_layer),
	number_of_hidden_layers(number_of_hidden_layers), net_number_of_neurons(net_number_of_neurons),
	number_of_features(network_number_of_features),

	// allocate memory to the best pointers
	best_weights(allocate_memory_for_weights(number_of_neurons_each_hidden_layers, number_of_hidden_layers, number_of_features)),
	best_biases(allocate_memory_for_biases(number_of_neurons_each_hidden_layers, number_of_hidden_layers)),
	best_means_and_variances(allocate_memory_for_mv_or_ss(net_number_of_neurons)),
	best_scales_and_shifts(allocate_memory_for_mv_or_ss(net_number_of_neurons))
{ }

// deallocate all the best pointers
NeuralNetwork::BestStateLoader::~BestStateLoader()
{
	deallocate_memory_for_weights(best_weights, number_of_neurons_each_hidden_layer, number_of_hidden_layers);
	deallocate_memory_for_biases(best_biases, number_of_hidden_layers);
	deallocate_memory_for_mv_or_ss(best_means_and_variances, net_number_of_neurons);
	deallocate_memory_for_mv_or_ss(best_scales_and_shifts, net_number_of_neurons);
}

// update the best state to the current state of the neural network
void NeuralNetwork::BestStateLoader::save_best_state()
{
	write_to_best_weights();

	write_to_best_biases();

	write_to_best_means_and_variances();

	write_to_best_scales_and_shifts();
}

// copy the values of the current weights to the best weights pointer
void NeuralNetwork::BestStateLoader::write_to_best_weights()
{
	// for the first layer with features
	for (int n = 0; number_of_neurons_each_hidden_layer[0]; n++)
		for (int f = 0; f < number_of_features; f++)
			best_weights[0][n][f] = current_weights[0][n][f];

	// copy each hidden layers' weights inot the best weights
	for (int l = 1; l < number_of_hidden_layers; l++)
		for (int n = 0; number_of_neurons_each_hidden_layer[l]; n++)
			for (int f = 0; f < number_of_neurons_each_hidden_layer[l - 1]; f++)
				best_weights[l][n][f] = current_weights[l][n][f];

	// copy output layers' weights into the best weights
	for (int f = 0; f < number_of_neurons_each_hidden_layer[number_of_hidden_layers - 1]; f++)
		best_weights[number_of_hidden_layers][0][f] = current_weights[number_of_hidden_layers][0][f];
}

// copy the values of the current biases to the best biases pointer
void NeuralNetwork::BestStateLoader::write_to_best_biases()
{
	// copy each layers' biases into the best biases
	for (int l = 0; l < number_of_hidden_layers; l++)
		for (int n = 0; n < number_of_neurons_each_hidden_layer[l]; n++)
			best_biases[l][n] = current_biases[l][n];

	// copy the output layer's bias into the best bias
	best_biases[number_of_hidden_layers][0] = current_biases[number_of_hidden_layers][0];
}

// copy the values of the current running means and variances to the best means and variances pointer
void NeuralNetwork::BestStateLoader::write_to_best_means_and_variances()
{
	for (int n = 0; n < net_number_of_neurons; n++)
		for (int f = 0; f < 2; f++)
			best_means_and_variances[n][f] = current_means_and_variances[n][f];
}

// copy the values of the current scales and shifts to the scales and shifts pointer
void NeuralNetwork::BestStateLoader::write_to_best_scales_and_shifts()
{
	for (int n = 0; n < net_number_of_neurons; n++)
		for (int f = 0; f < 2; f++)
			best_scales_and_shifts[n][f] = current_scales_and_shifts[n][f];
}

// update the current state of the neural network to the best state
void NeuralNetwork::BestStateLoader::load_best_state()
{
	write_to_current_weights();

	write_to_current_biases();

	write_to_current_means_and_variances();

	write_to_current_scales_and_shifts();
}

// copy the values of the best weights into the current weights
void NeuralNetwork::BestStateLoader::write_to_current_weights()
{
	// for the first layer with features
	for (int n = 0; number_of_neurons_each_hidden_layer[0]; n++)
		for (int f = 0; f < number_of_features; f++)
			current_weights[0][n][f] = best_weights[0][n][f];

	// copy each hidden layers' weights inot the best weights
	for (int l = 1; l < number_of_hidden_layers; l++)
		for (int n = 0; number_of_neurons_each_hidden_layer[l]; n++)
			for (int f = 0; f < number_of_neurons_each_hidden_layer[l - 1]; f++)
				current_weights[l][n][f] = best_weights[l][n][f];

	// copy output layers' weights into the best weights
	for (int f = 0; f < number_of_neurons_each_hidden_layer[number_of_hidden_layers - 1]; f++)
		current_weights[number_of_hidden_layers][0][f] = best_weights[number_of_hidden_layers][0][f];
}

// copy the values of the best biases into the current biases
void NeuralNetwork::BestStateLoader::write_to_current_biases()
{
	// copy the best biases into the layers' biases
	for (int l = 0; l < number_of_hidden_layers; l++)
		for (int n = 0; n < number_of_neurons_each_hidden_layer[l]; n++)
			current_biases[l][n] = best_biases[l][n];

	// copy the output layer's best bias into the output layer's bias
	current_biases[number_of_hidden_layers][0] = best_biases[number_of_hidden_layers][0];
}

// copy the values of the best means and variances into the current means and variances
void NeuralNetwork::BestStateLoader::write_to_current_means_and_variances()
{
	for (int n = 0; n < net_number_of_neurons; n++)
		for (int f = 0; f < 2; f++)
			current_means_and_variances[n][f] = best_means_and_variances[n][f];
}

// copy the values of the best scales and shifts into the current scales and shifts
void NeuralNetwork::BestStateLoader::write_to_current_scales_and_shifts()
{
	for (int n = 0; n < net_number_of_neurons; n++)
		for (int f = 0; f < 2; f++)
			current_scales_and_shifts[n][f] = best_scales_and_shifts[n][f];
}