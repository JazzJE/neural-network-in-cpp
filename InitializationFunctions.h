#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <string>

// weights and biases file methods

	// generating weights and biases file
void generate_weights_and_biases_file(std::string weights_and_biases_file_name, const int* number_of_neurons_each_hidden_layer,
	int number_of_hidden_layers, int number_of_features);
void generate_weights_and_biases_for_layer(std::fstream& weights_and_biases_file, int number_of_features, int number_of_neurons);

	// parsing weights and biases file
void parse_weights_and_biases_file(std::fstream& weights_and_biases_file, double*** weights, double** biases,
	const int* number_of_neurons_each_hidden_layer, int number_of_hidden_layers, int number_of_features);
void parse_weights_and_biases_for_layer(std::fstream& weights_and_biases_file, int number_of_features, int number_of_neurons,
	double*** weights, double** biases, int layer_index);

	// validating weights and biases file
void validate_weights_and_biases_file(std::fstream& weights_and_biases_file, std::string weights_and_biases_file_name,
	const int* number_of_neurons_each_hidden_layer, int number_of_hidden_layers, int number_of_features);
int find_error_weights_and_biases_file(std::fstream& weights_and_biases_file, const int* number_of_neurons_each_hidden_layer,
	int number_of_hidden_layers, int number_of_features);
bool check_line_weights_and_biases_file(std::fstream& weights_and_biases_file, int number_of_features);

// dataset file methods

	// parsing dataset file
void parse_dataset_file(std::fstream& dataset_file, double** training_features, double* target_values, std::string* feature_names,
	std::string& target_name, int number_of_features, int number_of_samples);

	// validating dataset file
void validate_dataset_file(std::fstream& dataset_file, std::string dataset_file_name, int number_of_features);
int find_error_dataset_file(std::fstream& dataset_file, int number_of_features);

// miscallaneous methods
void randomize_training_samples(double** training_features, double* target_values, int number_of_samples);
int count_number_of_samples(std::fstream& dataset_file);
int count_number_of_features(std::fstream& dataset_file);