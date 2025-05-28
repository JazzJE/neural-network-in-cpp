#pragma once

// memory allocation methods

double*** allocate_memory_for_weights(const int* number_of_neurons_each_hidden_layer, int number_of_hidden_layers, int number_of_features);
double** allocate_memory_for_biases(const int* number_of_neurons_each_hidden_layer, int number_of_hidden_layers, int number_of_features);
double** allocate_memory_for_training_samples(int number_of_samples, int number_of_features);
double* allocate_memory_for_target_values(int number_of_samples);