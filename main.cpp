/*
	Programmer name: Jedrick Espiritu
	Program name: vectorless_neural_network.cpp
	Version: 1.0.0
	Date: May 4, 2025

	**KEY THINGS TO NOTE WHEN INTERACTING WITH THIS PROGRAM**

	* when you want to generate a new neural network, delete the "weights.csv" file
	* this program is hard-written with a single neuron for output (meaning that you should only be predicting one value)
	* in the csv file that you will parse, make sure that the last column is the column that you want to predict (i.e., if
	  you want to predict house prices, then every other column will be the house features, and then the last column will
	  be the actual house prices)
	* the program uses stochastic gradient descent as I do not have the appropriate skillset to apply parallelization as of yet
*/

#include <iostream>
#include <fstream>
#include <cmath>
#include <random>
#include "NeuralNetwork.h"
#include "DenseLayer.h"
#include "Neuron.h"

void generate_weight_file(std::fstream&, int number_of_neurons_each_layer[], int number_of_layers, int number_of_features);
void parse_csv();
void normalize_features();

// driver for class
int main()
{
	// !!! NOTE that the last column is the column of values you want to predict !!!
	// for example, if you have features of a house and want to predict price, then make the house prices the last column
	// of the csv that you want to use as your database
	// data set used: https://www.kaggle.com/datasets/meharshanali/amazon-stocks-2025	
	int number_of_samples = 6988;
	int number_of_features = 6;

	// !!! NOTE AGAIN that this program is hard-written with a single neuron for output !!!
	// this is the order and number of neurons you want in each hidden layer
	// in the below example...
		// the first hidden layer will have 5 neurons
		// the second hidden layer will have 10 neurons
		// the third hidden layer will have 3 neurons
	int number_of_neurons_each_layer[] = { 5, 10, 3 };

	// change this value to change how "fast" the neural network learns your data
	double learning_rate = 0.01;

	// change this value to regularize and prevent overfitting for the neural network
	double regularization = 0.01;

	// open the file for weights
	std::fstream weight_file("./weights.csv", std::ios::out | std::ios::in);
	// if the weight file was not opened (therefore doesn't exist), initialize a new one using He initialization
	if (!weight_file)
	{
		generate_weight_file(weight_file, number_of_neurons_each_layer, 
			sizeof(number_of_neurons_each_layer) / sizeof(int), number_of_features);
		weight_file.open("./weights.csv", std::ios::out | std::ios::in);
	}
}

// generate weight file if not already made
void generate_weight_file(std::fstream& weight_file, int number_of_neurons_each_layer[], int number_of_layers, int number_of_features)
{
	// create a new weight file
	weight_file.open("./weights.csv", std::ios::out);

	// generate a random seed for random numbers
	std::random_device rd;
	std::mt19937 gen(rd());

	// use He initialization for the weights provided the number of input features into the first layer
	double stddev = sqrt(2.0 / number_of_features);
	std::normal_distribution<double> dist(0.0, stddev);
	// for each neuron in the first layer
	for (int n = 0; n < number_of_neurons_each_layer[0]; n++)
	{
		// insert an equal number of weights into the 
		for (int w = 0; w < number_of_features; w++)
			weight_file << dist(gen) << ",";

		// insert an initial bias value of 0 and then end the current neuron line
		weight_file << 0 << '\n';
	}

	// now generate initial weights using He initialization for every subsequent layer and store in weight file
	for (int l = 1; l < number_of_layers; l++)
	{
		// create a random number generator with He initialization
		// the number of features for a given layer l is the l - 1 number of neurons because these are dense layers
		stddev = sqrt(2.0 / number_of_neurons_each_layer[l - 1]);
		dist.param(std::normal_distribution<double>::param_type(0.0, stddev));

		// for each neuron in current the layer, initialize weights for each neuron
		for (int n = 0; n < number_of_neurons_each_layer[l]; n++)
		{
			for (int w = 0; w < number_of_neurons_each_layer[l - 1]; w++)
				weight_file << dist(gen) << ",";

			// insert an initial bias value of 0 and then end the current neuron line
			weight_file << 0 << '\n';
		}
	}

	// close the new file so it can be reopened with new settings
	weight_file.close();
}

// parse in a csv of data
void parse_csv()
{

}