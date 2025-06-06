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
#include <iomanip>
#include <fstream>
#include <random>
#include <cstdlib>
#include <string>
#include "InitializationFunctions.h"
#include "MemoryFunctions.h"
#include "MenuFunctions.h"
#include "DenseLayer.h"
#include "NeuralNetwork.h"
#include "Neuron.h"
#include "StatisticsFunctions.h"

// driver for class
int main()
{
	// !!! REMINDER that the last column is the column of values you want to predict !!!
		// for example, if you have features of a house and want to predict price, then make the house prices the last column
		// of the csv that you want to use as your database
	// data set used: https://www.kaggle.com/datasets/meharshanali/amazon-stocks-2025	

	// !!! NOTE AGAIN that this program is hard-written with a single neuron for output !!!
	// this is the order and number of neurons you want in each hidden layer
	// in the below example...
		// the first hidden layer will have 5 neurons
		// the second hidden layer will have 10 neurons
		// the third hidden layer will have 3 neurons
		// the fourth/output layer will have 1 neuron, predicting the value
	const int number_of_neurons_each_hidden_layer[] = { 7, 10, 3 };



	// PAST THIS POINT IS ALL THE HARD CODE; REFER TO ABOVE PARTS FOR EDITABLE COMPONENTS
	
	std::cout << "The program may take a couple of seconds to load...\n" << std::fixed;

	// initialize new seed
	srand(time(0));

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

	// ensure that the dataset file has the correct number of features for each line, and if not, then end the program
	validate_dataset_file(dataset_file, dataset_file_name, number_of_features);

	// allocate memory for the training samples and target values
	double** training_features = allocate_memory_for_training_features(number_of_samples, number_of_features);
	double* target_values = new double[number_of_samples];

	// parse the csv dataset into the 2d training samples array, but also get the names of the columns
	std::string* feature_names = new std::string[number_of_features];
	std::string target_name;
	parse_dataset_file(dataset_file, training_features, target_values, feature_names, target_name, number_of_features, number_of_samples);

	// randomize the training samples orders a lot to ensure randomness
	int number_of_shuffles = 5;
	for (int s = 0; s < number_of_shuffles; s++)
		randomize_training_samples(training_features, target_values, number_of_samples);

	// calculate the means and standard deviations of all the features and normalize them
	double* all_samples_means = calculate_features_means(training_features, number_of_features, number_of_samples);
	double* all_samples_stddevs = calculate_features_stddevs(training_features, all_samples_means, number_of_features, number_of_samples);
	double** all_features_normalized = calculate_normalized_features(training_features, number_of_samples, number_of_features, all_samples_means, 
		all_samples_stddevs);

	// close dataset file when done
	dataset_file.close();


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

	// ensure that the weights and biases file has the appropriate weights and biases for each layer, else prompt the user 
	// to make a new file
	validate_weights_and_biases_file(weights_and_biases_file, weights_and_biases_file_name, number_of_neurons_each_hidden_layer,
		number_of_hidden_layers, number_of_features);

	// 3d array to hold all the weights and biases of each neuron of each layer 
	// add one to account for the output layer
	double*** const weights = allocate_memory_for_weights(number_of_neurons_each_hidden_layer, number_of_hidden_layers, number_of_features);
	double** const biases = allocate_memory_for_biases(number_of_neurons_each_hidden_layer, number_of_hidden_layers);

	// parse the weights and biases into a 3d array
	parse_weights_and_biases_file(weights_and_biases_file, weights, biases, 
		number_of_neurons_each_hidden_layer, number_of_hidden_layers, number_of_features);

	// close weights file when done
	weights_and_biases_file.close();

	// begin menus and actual interaction with neural network from this line onwards
	char option;
	double learning_rate, regularization_rate;

	// ask user for an initial value of the learning rate and the regularization values
	std::cout << "Hello! Welcome to my hard-coded neural network program.\n";
	std::cout << "\nBefore beginning, please give initial values for the following parameters.\n\n";

	generate_border_line();
	input_parameter_rates(learning_rate, regularization_rate);
	generate_border_line();

	// create the neural network
	NeuralNetwork neural_network(weights, biases, number_of_neurons_each_hidden_layer, number_of_hidden_layers,
		number_of_features, learning_rate, regularization_rate);

	while (true)
	{
		std::cout << "\nOption Menu:"
			<< "\n\t1. Train neural network (five-fold, mini-batch gradient descent)"
			<< "\n\t2. Predict a value"
			<< "\n\t3. Save your current neural network (update the weights and biases file)"
			<< "\n\t4. Change learning and regularization parameters"
			<< "\n\t5. Exit program (exiting will not save the network)"
			<< "\nPlease select an option: ";
		std::cin >> option;

		// input validation
		while (option < '1' || option > '5')
		{
			std::cout << "[ERROR] Please enter a valid input (1-5): ";
			std::cin >> option;
		}

		// end the program if selected
		if (option == '5') break;

		generate_border_line();

		switch (option)
		{
		case '1': // train neural network

			std::cout << "\n\tTraining your network...";
			neural_network.five_fold_train(training_features, target_values, number_of_samples);
			std::cout << "\n\tDone!\n";

			break; // end case

		case '2': // predict a value

		{
			int random_index = std::rand() % number_of_samples;
			std::cout << "\nProvided these features for sample #" << random_index << " : ";
			for (int f = 0; f < number_of_features; f++)
				std::cout << "\n\t" << feature_names[f] << " - " << training_features[random_index][f];

			std::cout << "\n\nActual value of " << target_name << ": " << target_values[random_index];

			std::cout << "\n\nPrediction of " << target_name << ": " << neural_network.calculate_prediction(training_features[random_index]) << "\n";

			break; // end case
		}

		case '3': // save current neural network

			update_weights_and_biases_file(weights_and_biases_file_name, weights, biases, 
				number_of_neurons_each_hidden_layer, number_of_hidden_layers, number_of_features);

			break; // end case

		case '4': // change learning and regularization parameters

			input_parameter_rates(learning_rate, regularization_rate);
			neural_network.set_learning_rate(learning_rate);
			neural_network.set_regularization_rate(regularization_rate);

			// end case

		}

		generate_border_line();

	}

	std::cout << "\nEnding program...\n";
	return 0;

}
