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
#include <cmath>
#include <string>
#include "InitializationFunctions.h"
#include "NeuralNetwork.h"
#include "DenseLayer.h"
#include "Neuron.h"

// driver for class
int main()
{
	// !!! REMINDER that the last column is the column of values you want to predict !!!
		// for example, if you have features of a house and want to predict price, then make the house prices the last column
		// of the csv that you want to use as your database
	// !!! ALSO !!! the target values column does not count towards the number of features (i.e., if you have only 2 features 
	// and are predicting house prices, just put number_of_features as 2)


	// data set used: https://www.kaggle.com/datasets/meharshanali/amazon-stocks-2025	
	// order of columns
		// 1. Time since Unix epoch
		// 2. Opening price
		// 3. Highest price
		// 4. Lowest Price

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
	

	// open the file for the samples
	std::string dataset_file_name = "dataset.csv";
	std::fstream dataset_file(dataset_file_name, std::ios::in);

	// if dataset could not be found, then just output an error
	if (!dataset_file) 
	{
		std::cerr << "[ERROR] The dataset could not be found within the project; please edit the \"dataset_file_name\" variable "
			<< "to the dataset's name, or otherwise include the dataset within the project." << std::endl;
		exit(0);
	}

	// calculate number of samples, features, and number of hidden layers
	int number_of_samples = count_number_of_samples(dataset_file);
	int number_of_features = count_number_of_features(dataset_file);
	int number_of_hidden_layers = sizeof(number_of_neurons_each_hidden_layer) / sizeof(int);

	// output an error and end program if the samples are not consistent/do not all have the same number of columns/features
	// the find error dataset file will return an integer value of the line in which the error was found
	// if no error is found, the line error is set to -1
	int line_error = find_error_dataset_file(dataset_file, number_of_features);
	if (line_error != -1)
	{
		std::cerr << "[ERROR] The dataset is inconsistent; some rows have more features/columns than others."
			<< "\n\n\t*** The error was found on line #" << line_error << " in " << dataset_file_name << " ***";

		std::cout << "\n\nEnding program...\n";
		exit(0);
	}

	double** training_samples = new double* [number_of_samples];
	for (int i = 0; i < number_of_samples; i++)
		training_samples[i] = new double[number_of_features];
	
	double* target_values = new double[number_of_samples];

	// parse the csv dataset into the 2d training samples array
	parse_csv_sample_data(dataset_file, training_samples, target_values, number_of_features, number_of_samples);

	// randomize the training samples orders a lot to ensure randomness
	int number_of_shuffles = 5;
	for (int s = 0; s < number_of_shuffles; s++)
		randomize_training_samples(training_samples, number_of_samples);


	// open the file for weights
	std::string weights_and_biases_file_name = "weights_and_biases.csv";
	std::fstream weights_and_biases_file(weights_and_biases_file_name, std::ios::out | std::ios::in);

	// if the weight file was not opened (therefore doesn't exist), initialize a new one using He initialization
	if (!weights_and_biases_file)
	{
		std::cout << "Weights and biases file not found; creating new one...";
		
		// create a new weight file
		generate_weights_and_biases_file(weights_and_biases_file_name, number_of_neurons_each_hidden_layer, 
			number_of_hidden_layers, number_of_features);

		weights_and_biases_file.open(weights_and_biases_file_name, std::ios::out | std::ios::in);
	}

	// validate the weights and biases file is valid
	// the find error dataset file will return an integer value of the line in which the error was found
	// if no error is found, the line error is set to -1
	line_error = find_error_weights_and_biases_file(weights_and_biases_file, number_of_neurons_each_hidden_layer, number_of_hidden_layers, number_of_features);
	if (line_error != - 1)
	{
		char option;

		// ask user if they would like to reset their neural network, and if not, then end the program
		// so they can update the configuration to their weights and biases folder
		std::cerr << "\nWeights and biases file is erroneous (there are not enough weights to accomodate all features)"
			<< "\n\n\t*** The error was found on line #" << line_error << " in " << weights_and_biases_file_name << " ***"
			<< "\n\nWould you like to generate a new weights and biases file?"
			<< "\n\n******************WARNING******************"
			<< "\n\nThis is effective to resetting your neural network, and should only be done when changing the number of features "
			<< "or changing the number of neurons in the hidden layers; if you choose no, the program will end such that"
			<< "you can update the number_of_features and number_of_neurons_in_each_hidden_layer array to the correct configuration"
			<< "\n\n*******************************************"
			<< "\n\nPlease select yes or no (Y/N): ";
		std::cin >> option;

		while (option != 'Y' && option != 'N')
		{
			std::cout << "[ERROR] Invalid input. Please enter only yes or no (Y/N): ";
			std::cin >> option;
		}

		if (option == 'Y')
		{
			weights_and_biases_file.close();

			generate_weights_and_biases_file(weights_and_biases_file_name, number_of_neurons_each_hidden_layer,
				sizeof(number_of_neurons_each_hidden_layer) / sizeof(int), number_of_features);

			weights_and_biases_file.open(weights_and_biases_file_name, std::ios::out | std::ios::in);
		}
		else
		{
			std::cout << "\nEnding program...\n";
			exit(0);
		}
	}

	// parse the weights and biases into a 3d array now





}
