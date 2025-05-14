#ifndef INITIALIZATIONFUNCTIONS_H
#define INITIALIZATIONFUNCTIONS_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <random>
#include <string>

int count_number_of_samples(std::fstream& dataset_file);
int count_number_of_features(std::fstream& dataset_file);

void generate_weights_and_biases_file(const std::string& weights_and_biases_file_name, const int* number_of_neurons_each_hidden_layer,
	const int& number_of_hidden_layers, const int& number_of_features);
void generate_weights_and_biases_for_layer(std::fstream& weights_and_biases_file, int number_of_features, int number_of_neurons);

void parse_csv_sample_data(std::fstream& dataset_file, double**& training_samples, double* target_values,
	const int& number_of_features, const int& number_of_samples);
void randomize_training_samples(double**& training_samples, const int& number_of_samples);

int find_error_dataset_file(std::fstream& dataset_file, const int& number_of_features);
int find_error_weights_and_biases_file(std::fstream& weights_and_biases_file, int* number_of_neurons_each_hidden_layer,
	const int& number_of_hidden_layers, const int& number_of_features);
bool validate_weights_and_biases_for_line(std::fstream& weights_and_biases_file, int number_of_features);

#endif
