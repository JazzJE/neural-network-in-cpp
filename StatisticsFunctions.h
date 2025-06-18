#pragma once
#include "MemoryFunctions.h"
#include <cmath>

double* calculate_features_means(double** sample_features, int number_of_features, int number_of_samples,
	int lower_validation_index = -1, int higher_validation_index = -1);
double* calculate_features_stddevs(double** sample_features, double* features_means, int number_of_features,
	int number_of_samples, int lower_validation_index = -1, int higher_validation_index = -1);
double** calculate_normalized_features(double** sample_features, int number_of_samples, int number_of_features, double* means_array, double* stddevs_array);
double* calculate_normalized_features(double* sample_features, int number_of_features, double* means_array, double* stddevs_array);