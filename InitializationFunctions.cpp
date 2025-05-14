#include "InitializationFunctions.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <random>
#include <string>

// count the number of samples in the file
int count_number_of_samples(std::fstream& dataset_file)
{
	int counter = 0;
	std::string line;

	while (getline(dataset_file, line))
		counter++;

	// reset to start
	dataset_file.clear();
	dataset_file.seekg(0);

	return counter;
}

// count number of features by taking in a line
// note that the number of features is equal to the number of fields taken in minus 1, given last column is training sample
int count_number_of_features(std::fstream& dataset_file)
{
	int counter = 0;
	std::string line, value;

	getline(dataset_file, line);
	std::stringstream ss(line);

	while (getline(ss, value, ','))
		counter++;

	// reset to start
	dataset_file.clear();
	dataset_file.seekg(0);

	return (counter - 1);
}

// generate weight file if not already made
void generate_weights_and_biases_file(const std::string& weights_and_biases_file_name, const int* number_of_neurons_each_hidden_layer,
	const int& number_of_hidden_layers, const int& number_of_features)
{
	// create the file
	std::fstream weights_and_biases_file(weights_and_biases_file_name, std::ios::out | std::ios::trunc);

	generate_weights_and_biases_for_layer(weights_and_biases_file, number_of_features, number_of_neurons_each_hidden_layer[0]);

	// now generate initial weights using He initialization for every subsequent layer and store in weight file
	for (int l = 1; l < number_of_hidden_layers; l++)
		generate_weights_and_biases_for_layer(weights_and_biases_file, number_of_neurons_each_hidden_layer[l - 1],
			number_of_neurons_each_hidden_layer[l]);

	generate_weights_and_biases_for_layer(weights_and_biases_file, number_of_neurons_each_hidden_layer[number_of_hidden_layers - 1], 1);

	// close the file
	weights_and_biases_file.close();
}

void generate_weights_and_biases_for_layer(std::fstream& weights_and_biases_file, int number_of_features, int number_of_neurons)
{
	// generate a random seed for numbers
	std::random_device rd;
	std::mt19937 gen(rd());

	// use He initialization for the weights provided the number of input features into the first layer
	double stddev = sqrt(2.0 / number_of_features);
	std::normal_distribution<double> dist(0.0, stddev);
	// for each neuron in the first layer
	for (int n = 0; n < number_of_neurons; n++)
	{
		// insert an equal number of weights into the 
		for (int w = 0; w < number_of_features; w++)
			weights_and_biases_file << dist(gen) << ",";

		// insert an initial bias value of 0 and then end the current neuron line
		weights_and_biases_file << 0 << '\n';
	}
}

// validate that the samples are valid (all of them have the same amount of features)
// the function will return the line in which the error was found so it can be altered easily; if not found, return -1
int find_error_dataset_file(std::fstream& dataset_file, const int& number_of_features)
{
	std::string line, value;
	std::stringstream ss;

	// this will store the line number which the error was found so user may alter it accordingly
	int line_error = 0;

	// store number of fields for each line for validation
	int field_count = 0;

	while (getline(dataset_file, line))
	{
		line_error++;
		field_count = 0;

		// reset the stringstream
		ss.clear();
		ss.str(line);

		while (getline(ss, value, ',')) field_count++;

		if (field_count - 1 != number_of_features) return line_error;
	}

	// reset to start
	dataset_file.clear();
	dataset_file.seekg(0);
	
	// no error was found
	line_error = -1;

	return line_error;
}

// validate the file has the correct number of weights and bias values
// the function will return the line the error was found so it can be altered easily; if not found, return -1
int find_error_weights_and_biases_file(std::fstream& weights_and_biases_file, int* number_of_neurons_each_hidden_layer, const int& number_of_hidden_layers,
	const int& number_of_features)
{
	// this will store the line number which the error was found so user may alter it accordingly
	int line_error = 0;

	// for each neuron line n in the first layer, see if there is an equivalent number of features/weights for the number of features
	// plus one bias value
	for (int n = 0; n < number_of_neurons_each_hidden_layer[0]; n++)
	{
		line_error++;
		if (!validate_weights_and_biases_for_line(weights_and_biases_file, number_of_features)) return line_error;
	}

	// for each neuron line n in the given layer l, ensure there is an equivalent number of features/weights for the l-1th layer 
	// plus one bias value
	for (int l = 1; l < number_of_hidden_layers; l++)
		for (int n = 0; n < number_of_neurons_each_hidden_layer[l]; n++)
		{
			line_error++;
			if (!validate_weights_and_biases_for_line(weights_and_biases_file, number_of_neurons_each_hidden_layer[l - 1]))
				return line_error;
		}

	// for the last layer with one neuron, check if it has enough features for the last specific layer in the number of neurons each layer array
	// plus one bias value
	line_error++;
	if (!validate_weights_and_biases_for_line(weights_and_biases_file, number_of_neurons_each_hidden_layer[number_of_hidden_layers - 1]))
		return line_error;

	// if reached this point, means all the lines are fine, and thus reset to start of the file
	weights_and_biases_file.seekg(0);

	// no error was found
	line_error = -1;

	return line_error;
}

bool validate_weights_and_biases_for_line(std::fstream& weights_and_biases_file, int number_of_features)
{
	std::string line, value;

	getline(weights_and_biases_file, line);

	std::stringstream ss(line);

	// count the number of weights and biases in a given row
	// for a given row, there should be a number of features + one to simulate
	// how each neuron will have an equal number of weights + one bias value
	int field_count = 0;
	while (getline(ss, value, ','))
		field_count++;

	if (field_count != number_of_features + 1) return false;

	return true;
}


// parse in the csv of data
void parse_csv_sample_data(std::fstream& dataset_file, double**& training_samples, double* target_values,
	const int& number_of_features, const int& number_of_samples)
{
	std::string line;

	// for each training sample t
	for (int t = 0; t < number_of_samples; t++)
	{
		// get the line of features
		getline(dataset_file, line);

		std::stringstream ss(line);
		std::string value;

		// get each feature and input them into the training samples
		for (int f = 0; f < number_of_features; f++)
		{
			getline(ss, value, ',');
			training_samples[t][f] = std::stod(value);
		}

		// get the target value, which is the last column
		getline(ss, value, '\n');
		target_values[t] = std::stod(value);
	}

	dataset_file.close();

}

// randomize the order of the training samples
void randomize_training_samples(double**& training_samples, const int& number_of_samples)
{
	int random_index;
	double* temp;

	for (int current_index = number_of_samples - 1; current_index > 0; current_index--)
	{
		random_index = std::rand() % current_index;

		// swap where pointers are directed
		temp = training_samples[random_index];
		training_samples[random_index] = training_samples[current_index];
		training_samples[current_index] = temp;
	}
}