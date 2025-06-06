#include "InitializationFunctions.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <random>
#include <string>

// generating weights and biases file methods

	// generate weight file if not already made
void generate_weights_and_biases_file(std::string weights_and_biases_file_name, const int* number_of_neurons_each_hidden_layer,
	int number_of_hidden_layers, int number_of_features)
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

	// helper function to generate the weights and biases for a given layer with a set number of neurons
	// did this because updating the normal distribution
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

	// parse the weights_and_biases file into an array
void parse_weights_and_biases_file(std::fstream& weights_and_biases_file, double*** weights, double** biases,
	const int* number_of_neurons_each_hidden_layer, int number_of_hidden_layers, int number_of_features)
{
	// parse weights and biases for first layer
	parse_weights_and_biases_for_layer(weights_and_biases_file, number_of_features, 
		number_of_neurons_each_hidden_layer[0], weights, biases, 0);

	// parse the rest of the layers into the weights and biases array
	for (int l = 1; l < number_of_hidden_layers; l++)
		parse_weights_and_biases_for_layer(weights_and_biases_file, number_of_neurons_each_hidden_layer[l - 1],
			number_of_neurons_each_hidden_layer[l], weights, biases, l);

	// parse last layer into the weights and biases array
	// remember that the number of hidden layers is equal to the index of the last layer
	parse_weights_and_biases_for_layer(weights_and_biases_file, number_of_neurons_each_hidden_layer[number_of_hidden_layers - 1],
		1, weights, biases, number_of_hidden_layers);

}

	// helper function to parse each neuron of a given layer
void parse_weights_and_biases_for_layer(std::fstream& weights_and_biases_file, int number_of_features, int number_of_neurons, 
	double***weights, double** biases, int layer_index)
{
	std::string line, value;
	std::stringstream ss;
	
	for (int n = 0; n < number_of_neurons; n++)
	{
		getline(weights_and_biases_file, line);

		ss.clear();
		ss.str(line);

		for (int w = 0; w < number_of_features; w++)
		{
			getline(ss, value, ',');
			weights[layer_index][n][w] = std::stod(value);
		}

		// last value will be bias value
		getline(ss, value, '\n');
		biases[layer_index][n] = std::stod(value);
	}
}

// methods to validate the weights and biases file

	// validate the weights and biases file is valid, and if not, prompt user to make a new one
void validate_weights_and_biases_file(std::fstream& weights_and_biases_file, std::string weights_and_biases_file_name, 
	const int* number_of_neurons_each_hidden_layer, int number_of_hidden_layers, int number_of_features)
{
	// the find_error_weights_and_biases_file function will return an integer value of the line in which the error was found
	// if no error is found, the error returned is -1
	int line_error = find_error_weights_and_biases_file(weights_and_biases_file, number_of_neurons_each_hidden_layer,
		number_of_hidden_layers, number_of_features);
	if (line_error != -1)
	{
		char option;

		// ask user if they would like to reset their neural network, and if not, then end the program
		// this is so they can update the configuration to their weights and biases folder if they accidentally interacted with it
		std::cerr << "Weights and biases file is erroneous (there are not enough weights to accomodate all features "
			<< "OR a string value was detected [AFTER THE FIRST LINE, there should only be double values])"
			<< "\n\n\t*** The error was found on line #" << line_error << " in " << weights_and_biases_file_name << " ***"
			<< "\n\nWould you like to generate a new weights and biases file?"
			<< "\n\n******************WARNING******************"
			<< "\n\nThis is effective to resetting your neural network, and should really only be done when changing the number of features "
			<< "or changing the number of neurons in the hidden layers; if you choose no, the program will end, such that"
			<< "you can update the number_of_features and number_of_neurons_in_each_hidden_layer array to the correct configuration, "
			<< "or fix the weights and biases file to not have any string values"
			<< "\n\n*******************************************"
			<< "\n\nPlease select yes or no (Y/N): ";
		std::cin >> option;

		while (option != 'Y' && option != 'N')
		{
			std::cerr << "[ERROR] Invalid input. Please enter only yes or no (Y/N): ";
			std::cin >> option;
		}

		if (option == 'Y')
		{
			weights_and_biases_file.close();

			generate_weights_and_biases_file(weights_and_biases_file_name, number_of_neurons_each_hidden_layer,
				number_of_hidden_layers, number_of_features);

			weights_and_biases_file.open(weights_and_biases_file_name, std::ios::out | std::ios::in);
		}
		else
		{
			std::cerr << "\nEnding program...\n";
			exit(0);
		}
	}
}

	// the function will return the line the error was found so it can be altered easily; if not found, return -1
int find_error_weights_and_biases_file(std::fstream& weights_and_biases_file, const int* number_of_neurons_each_hidden_layer, 
	int number_of_hidden_layers, int number_of_features)
{
	// this will store the line number which the error was found so user may alter it accordingly
	int line_error = 0;

	try
	{
		// for each neuron line n in the first layer, see if there is an equivalent number of features/weights for the number of features
		// plus one bias value
		for (int n = 0; n < number_of_neurons_each_hidden_layer[0]; n++)
		{
			line_error++;
			if (!check_line_weights_and_biases_file(weights_and_biases_file, number_of_features)) return line_error;
		}

		// for each neuron line n in the given layer l, ensure there is an equivalent number of features/weights for the l-1th layer 
		// plus one bias value
		for (int l = 1; l < number_of_hidden_layers; l++)
			for (int n = 0; n < number_of_neurons_each_hidden_layer[l]; n++)
			{
				line_error++;
				if (!check_line_weights_and_biases_file(weights_and_biases_file, number_of_neurons_each_hidden_layer[l - 1]))
					return line_error;
			}

		// for the last layer with one neuron, check if it has enough features for the last specific layer in the number of neurons each layer array
		// plus one bias value
		line_error++;
		if (!check_line_weights_and_biases_file(weights_and_biases_file, number_of_neurons_each_hidden_layer[number_of_hidden_layers - 1]))
			return line_error;

		// if reached this point, means all the lines are fine, and thus reset to start of the file
		weights_and_biases_file.seekg(0);
	}

	// return the line in which the string value was detected
	catch (std::invalid_argument)
	{
		return line_error;
	}

	// no error was found
	line_error = -1;

	return line_error;
}

	// check the specific given line of the weights and biases file is valid
bool check_line_weights_and_biases_file(std::fstream& weights_and_biases_file, int number_of_features)
{
	std::string line, value;

	getline(weights_and_biases_file, line);

	std::stringstream ss(line);

	// for a given row, there should be a number of features + one to simulate how each neuron will have an equal 
	// number of weights + one bias value
	int field_count = 0;
	while (getline(ss, value, ','))
	{
		// try turning each value parsed into a double, and if it fails, that means it's a string value, and therefore throw an error
		// there should only be numbers in the weights_and_biases_file
		std::stod(value);

		field_count++;
	}

	if (field_count != number_of_features + 1) return false;

	return true;
}

// dataset file methods

	// parse in the csv of data
void parse_dataset_file(std::fstream& dataset_file, double** training_samples, double* target_values, std::string* feature_names,
	std::string& target_name, int number_of_features, int number_of_samples)
{
	std::string line, value;
	std::stringstream ss;
	
	// get the first line with titles and parse them into the column names array
	getline(dataset_file, line);

	ss.clear();
	ss.str(line);

	for (int f = 0; f < number_of_features; f++)
	{
		getline(ss, value, ',');
		feature_names[f] = value;
	}

	getline(ss, value, '\n');
	target_name = value;


	// for each training sample t
	for (int t = 0; t < number_of_samples; t++)
	{
		// get the line of features
		getline(dataset_file, line);

		// clear stringstream object
		ss.clear();
		ss.str(line);

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
}

	// output an error and end program if the samples are not consistent/do not all have the same number of columns/features
void validate_dataset_file(std::fstream& dataset_file, std::string dataset_file_name, int number_of_features)
{
	// the find_error_dataset_file function will return an integer value of the line in which the error was found
	// if no error is found, the error returned is -1
	int line_error = find_error_dataset_file(dataset_file, number_of_features);
	if (line_error != -1)
	{
		std::cerr << "[ERROR] The dataset is inconsistent (aka some rows have more features/columns than others) "
			<< "OR there is a string value in the dataset(this program only accepts double values)."
			<< "\n\n\t*** The error was found on line #" << line_error << " in " << dataset_file_name << " ***"
			<< "\n\nPlease update your dataset file accordingly."
			<< "\n\nEnding program...\n";
		exit(0);
	}
}

// validate that the samples are valid (all of them have the same amount of features)
// the function will return the line in which the error was found so it can be altered easily; if not found, return -1
int find_error_dataset_file(std::fstream& dataset_file, int number_of_features)
{
	std::string line, value;
	std::stringstream ss;

	// this will store the line number which the error was found so user may alter it accordingly
	int line_error = 1;

	// store number of fields for each line for validation
	int field_count;

	// ignore the first line with column titles
	getline(dataset_file, line);

	try
	{
		while (getline(dataset_file, line))
		{
			line_error++;
			field_count = 0;

			// reset the stringstream
			ss.clear();
			ss.str(line);

			while (getline(ss, value, ','))
			{
				// validate that the value being parsed is not a string or anamolous � there should only be double values
				std::stod(value);

				field_count++;
			}

			// the field count also counts the last column, but must ignore it as it's not an input feature, but the target values
			if (field_count - 1 != number_of_features) return line_error;
		}
	}

	// if a string value was detected in the field, then return the line the string was found
	catch (std::invalid_argument)
	{
		return line_error;
	}

	// reset to start
	dataset_file.clear();
	dataset_file.seekg(0);

	// no error was found
	line_error = -1;

	return line_error;
}

// miscellaneous methods

	// randomize the order of the training samples
void randomize_training_samples(double** training_features, double* target_values, int number_of_samples)
{	
	int random_index;
	double temp_double;
	double* temp_ptr;

	for (int current_index = number_of_samples - 1; current_index > 0; current_index--)
	{
		random_index = std::rand() % current_index;

		// swap where pointers are directed
		temp_ptr = training_features[random_index];
		training_features[random_index] = training_features[current_index];
		training_features[current_index] = temp_ptr;

		// swap the target values
		temp_double = target_values[random_index];
		target_values[random_index] = target_values[current_index];
		target_values[current_index] = temp_double;
	}
}

	// count the number of samples in the file
int count_number_of_samples(std::fstream& dataset_file)
{
	int counter = 0;
	std::string line;

	// ignore first line with titles
	getline(dataset_file, line);

	while (getline(dataset_file, line))
		counter++;

	// reset to start
	dataset_file.clear();
	dataset_file.seekg(0);

	return counter;
}

	// count number of column titles, which is equal to number of features
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