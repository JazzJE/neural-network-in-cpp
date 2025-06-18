#pragma once
#include "DenseLayer.h"
#include "OutputLayer.h"
#include "MemoryFunctions.h"
#include "StatisticsFunctions.h"
#include <fstream>
#include <cstdlib>
#include <iostream>

class NeuralNetwork
{
private:

	// used for deleting network and training
	const int net_number_of_neurons;

	const int network_number_of_features;
	const int batch_size;

	const int* number_of_neurons_each_hidden_layer;
	const int number_of_hidden_layers;
	DenseLayer** const hidden_layers;
	OutputLayer* const output_layer;

	double* const network_regularization_rate;
	double* const network_learning_rate;

	double*** const network_weights;
	double** const network_biases;
	double** const network_means_and_variances;
	double** const network_scales_and_shifts;
	
	// this will save and write out to the neural network the best state of the program
	// only really will be used during training
	struct BestStateLoader
	{
		// store pointers to the network weights, biases, means & variances, and scales & shifts
		double*** const current_weights;
		double** const current_biases;
		double** const current_means_and_variances;
		double** const current_scales_and_shifts;
		// use these to interact with the pointers
		const int* number_of_neurons_each_hidden_layer;
		const int number_of_hidden_layers;
		const int net_number_of_neurons;
		const int number_of_features;

		// store pointers to the best states
		double*** const best_weights;
		double** const best_biases;
		double** const best_means_and_variances;
		double** const best_scales_and_shifts;

		BestStateLoader(double*** network_weights, double** network_biases, double** network_means_and_variances, double** scales_and_shifts,
			const int* number_of_neurons_each_hidden_layer, int number_of_hidden_layers, int net_number_of_neurons,
			int network_number_of_features);

		~BestStateLoader();

		// methods to save the best state
		void save_best_state();
		void write_to_best_weights();
		void write_to_best_biases();
		void write_to_best_means_and_variances();
		void write_to_best_scales_and_shifts();

		// methods to load best state
		void load_best_state();
		void write_to_current_weights();
		void write_to_current_biases();
		void write_to_current_means_and_variances();
		void write_to_current_scales_and_shifts();
	};

	void mini_batch_descent(BestStateLoader& bs_loader, double** training_features_normalized, double* target_values,
		int lower_cross_validation_index, int higher_cross_validation_index, int number_of_samples);

public:

	// initialize all the variables
	NeuralNetwork(double*** weights, double** biases, double** means_and_variances, double** scales_and_shifts, 
		const int* number_of_neurons_each_hidden_layer, int net_number_of_neurons, int number_of_hidden_layers, 
		int number_of_features, int batch_size, double learning_rate, double regularization_rate);

	// destructor to delete neural network
	~NeuralNetwork();

	// train the neural network five times based on the number of training samples
	void five_fold_train(double** training_features, double* target_values, int number_of_samples);

	// calculate a value based on the current weights and biases as well as the input features
	double calculate_prediction(double* input_features);

	// calculate the training predictions of each sample and return a dynamically allocated array to it
	double* calculate_training_predictions(double** normalized_input_features);

	// mutator/setter methods for rates
	void set_regularization_rate(double r_rate);
	void set_learning_rate(double l_rate);

};

