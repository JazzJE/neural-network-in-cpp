/*
	Programmer name: Jedrick Espiritu
	Program name: vectorless_neural_network.cpp
	Version: 1.0.0
	Date: May 4, 2025

	**KEY THINGS TO NOTE WHEN INTERACTING WITH THIS PROGRAM**

	- This program is hard-written with a single neuron for output (meaning that you should only be predicting one value)

	- In the csv dataset file that you will parse, make sure...
		
		1. That the last column is the column that you want to predict (i.e., if you want to predict house prices, then 
		every column except the last will be the house features, and then the last column will be the actual house prices)
		
		2. ***** THAT EVERY FIELD ONLY CONSISTS OF NUMBER VALUES *****; do not have strings/words/letters in the dataset 
		(i.e., do not have a column of strings like "name of house owner" and have every field underneath be strings like 
		"Jack Fry" or "Jill" or "ioenhaoea")

*/

#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <random>
#include <string>
#include <cctype>
#include "NeuralNetwork.h"
#include "DenseLayer.h"
#include "Neuron.h"

void generate_weights_and_biases_file(const std::string& weights_and_biases_file_name, const int* number_of_neurons_each_hidden_layer,
	const int& number_of_hidden_layers, const int& number_of_features);
void generate_weights_and_biases_for_layer(std::fstream& weights_and_biases_file, int number_of_features, int number_of_neurons);
void parse_csv_sample_data(const std::string& dataset_name, double** training_samples, double* target_values, 
	const int& number_of_features, const int& number_of_samples);
void calculate_standard_deviations(double* standard_deviations, const int& number_of_features);
void calculate_means(double* means, const int& number_of_features);
void normalize_input_features();
bool verify_weights_and_biases_file(std::fstream& weights_and_biases_file, int* number_of_neurons_each_hidden_layer, 
	int number_of_hidden_layers, const int& number_of_features);

// generate a random seed for random numbers
std::random_device rd;
std::mt19937 gen(rd());

// driver for class
int main()
{
	// !!! REMINDER that the last column is the column of values you want to predict !!!
		// for example, if you have features of a house and want to predict price, then make the house prices the last column
		// of the csv that you want to use as your database
	// !!! ALSO !!! the target values column does not count towards the number of features (i.e., if you have only 2 features 
	// and are predicting house prices, just put number_of_features as 2)


	// data set used: https://www.kaggle.com/datasets/meharshanali/amazon-stocks-2025	
	// order of columns: Date, Open, High, Low, Volume, Close
	int number_of_samples = 6987;
	int number_of_features = 5;


	// !!! NOTE AGAIN that this program is hard-written with a single neuron for output !!!
	// this is the order and number of neurons you want in each hidden layer
	// in the below example...
		// the first hidden layer will have 5 neurons
		// the second hidden layer will have 10 neurons
		// the third hidden layer will have 3 neurons
	int number_of_neurons_each_hidden_layer[] = { 7, 10, 3 };


	// change this value to change how "fast" the neural network learns your data
	double learning_rate = 0.01;


	// change this value to regularize and prevent overfitting for the neural network
	double regularization = 0.01;

	// PAST THIS POINT IS ALL THE HARD CODE; REFER TO ABOVE PARTS FOR EDITABLE COMPONENTS

	double* standard_deviations = new double[number_of_features];
	double* means = new double[number_of_features];

	double** training_samples = new double* [number_of_samples];

	// allocate memory to each array stored in the training samples
	for (int i = 0; i < number_of_samples; i++)
		training_samples[i] = new double[number_of_features];
	
	double* target_values = new double[number_of_samples];

	// parse the csv dataset into the 2d training samples array
	std::string dataset_name = "./dataset.csv";
	parse_csv_sample_data(dataset_name, training_samples, target_values, number_of_features, number_of_samples);


	// open the file for weights
	std::string weights_and_biases_file_name = "./weights_and_biases.csv";
	std::fstream weights_and_biases_file(weights_and_biases_file_name, std::ios::out | std::ios::in);

	// if the weight file was not opened (therefore doesn't exist), initialize a new one using He initialization
	if (!weights_and_biases_file)
	{
		std::cout << "Weights and biases file not found; creating new one...";
		
		// create a new weight file
		generate_weights_and_biases_file(weights_and_biases_file_name, number_of_neurons_each_hidden_layer, 
			sizeof(number_of_neurons_each_hidden_layer) / sizeof(int), number_of_features);

		weights_and_biases_file.open(weights_and_biases_file_name, std::ios::out | std::ios::in);
	}

	// verify the weights and biases file is valid
	if (!verify_weights_and_biases_file(weights_and_biases_file, number_of_neurons_each_hidden_layer, sizeof(number_of_neurons_each_hidden_layer) / sizeof(int), number_of_features))
	{
		char option;

		std::cout << "\nWeights and biases file is erroneous (there are not enough weights and biases to accomodate all features)"
			<< "\nWould you like to generate a new weights and biases file? (WARNING: this is effective "
			<< "\nto resetting your neural network, and should only be done when changing the number of features "
			<< "or changing the number of neurons in the hidden layers) (Y/N): ";
		std::cin >> option;

		while (tolower(option) != 'y' && tolower(option) != 'n')
		{
			std::cout << "[ERROR] Invalid input. Please enter only yes or no (Y/N): ";
			std::cin >> option;
		}

		if (tolower(option) == 'y')
		{
			weights_and_biases_file.close();

			generate_weights_and_biases_file(weights_and_biases_file_name, number_of_neurons_each_hidden_layer,
				sizeof(number_of_neurons_each_hidden_layer) / sizeof(int), number_of_features);

			weights_and_biases_file.open(weights_and_biases_file_name, std::ios::out | std::ios::in);
		}
	}
	
}

// verify the file has the correct number of weights and bias values
bool verify_weights_and_biases_file(std::fstream& weights_and_biases_file, int* number_of_neurons_each_hidden_layer, int number_of_hidden_layers,
	const int& number_of_features)
{
	// count the number of weights and biases in a given row
	// for a given row, there should be a number of features + one to simulate
	// how each neuron will have an equal number of weights + one bias value
	int field_count = 0;

	std::string line;
	getline(weights_and_biases_file, line);

	std::stringstream ss(line);
	std::string value;

	if (field_count != number_of_features + 1) return false;

	for (int l = 0; l < number_of_hidden_layers; l++)
	{
		// reset the field count for the line
		field_count = 0;

		// continue to the next layer's neurons
		for (int i = 0; i < number_of_neurons_each_hidden_layer[l] - 1; i++)
			getline(weights_and_biases_file, line);

		getline(weights_and_biases_file, line);
		ss.str(line);

		for (int f = 0; f < number_of_features; f++)
		{
			getline(ss, value, ',');
			field_count++;
		}

		if (field_count != number_of_neurons_each_hidden_layer[l] + 1) return false;
	}

	weights_and_biases_file.seekg(0); // reset to start
	return true;
}



// generate weight file if not already made
void generate_weights_and_biases_file(const std::string& weights_and_biases_file_name,  const int* number_of_neurons_each_hidden_layer, 
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

// parse in the csv of data
void parse_csv_sample_data(const std::string& dataset_name, double** training_samples, double* target_values, 
	const int& number_of_features, const int& number_of_samples)
{
	std::ifstream dataset_file(dataset_name, std::ios::in);

	if (!dataset_file.is_open()) {
		std::cerr << "[ERROR] The dataset could not be found within the project; please edit the \"dataset_name\" variable "
			<< "to the dataset's name, or otherwise include the dataset within the project." << std::endl;
	}
	else
	{

		std::string line;

		// for each training sample t
		for (int t = 0; t < number_of_samples; t++)
		{
			// get the line of features and target value
			getline(dataset_file, line);

			std::stringstream ss(line);
			std::string value;

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
}

// 
void calculate_standard_deviations(double* standard_deviations, const int& number_of_features)
{

}

// calculate the average of each feature
void calculate_means(double* means, const int& number_of_features)
{

}
