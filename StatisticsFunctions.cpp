#include "StatisticsFunctions.h"

// calculating means of a provided range

// the lower_validation_index and higher_validation_index parameters refer to the training sample of that is the 
// lower boundary of the cross-validation set and the higher boundary of the cross-validation set, such that the cross-validation
// set can be skipped over when calculating the means of the training set

double* calculate_features_means(double** sample_features, int number_of_features, int number_of_training_samples, 
	int lower_validation_index, int higher_validation_index)
{
	double* means_array = new double[number_of_features]();

	for (int t = 0; t < number_of_training_samples; t++)
	{
		// skip over the cross-validation set when calculating the features of the network
		if (t == lower_validation_index) t = higher_validation_index;
		
		else
			for (int f = 0; f < number_of_features; f++)
				means_array[f] += sample_features[t][f];
	}

	// the number of test samples is equal to the number of samples minus 1,
	// minus the difference between the high and low cross validation indices
	int number_of_test_samples = number_of_training_samples - 1 - (higher_validation_index - lower_validation_index);
	for (int f = 0; f < number_of_features; f++)
		means_array[f] /= number_of_test_samples;

	return means_array;
}

// calculating stddevs of a provided range

// the lower_validation_index and higher_validation_index parameters refer to the training sample of that is the 
// lower boundary of the cross-validation set and the higher boundary of the cross-validation set, such that the cross-validation
// set can be skipped over when calculating the stddevs of the training set

double* calculate_features_stddevs(double** sample_features, double* features_means, int number_of_features, 
	int number_of_training_samples, int lower_validation_index, int higher_validation_index)
{
	double* stddevs_array = new double[number_of_features]();

	for (int t = 0; t < number_of_training_samples; t++)
	{
		// skip over the cross-validation set when calculating the stddevs of the network
		if (t == lower_validation_index) t = higher_validation_index;

		else
			for (int f = 0; f < number_of_features; f++)
				stddevs_array[f] += pow(sample_features[t][f] - features_means[f], 2.0);
	}

	// the number of test samples is equal to the number of samples minus 1,
	// minus the difference between the high and low cross validation indices
	int number_of_test_samples = number_of_training_samples - 1 - (higher_validation_index - lower_validation_index);
	for (int f = 0; f < number_of_features; f++)
	{
		stddevs_array[f] /= number_of_test_samples;
		stddevs_array[f] = sqrt(stddevs_array[f]);
	}

	return stddevs_array;
}

// normalizing features in a new dynamically allocated array
// this is used for when given multiple samples of features
double** calculate_normalized_features(double** sample_features, int number_of_samples, int number_of_features, double* means_array, double* stddevs_array)
{
	double** normalized_features = allocate_memory_for_training_features(number_of_samples, number_of_features);

	for (int t = 0; t < number_of_samples; t++)
		for (int f = 0; f < number_of_features; f++)
			normalized_features[t][f] = ((sample_features[t][f] - means_array[f]) / sqrt(pow(stddevs_array[f], 2.0) + 1e-14));

	return normalized_features;
}

// this is used for when given a single sample of features
double* calculate_normalized_features(double* sample_features, int number_of_features, double* means_array, double* stddevs_array)
{
	double* normalized_features = new double[number_of_features];

	for (int f = 0; f < number_of_features; f++)
		normalized_features[f] = ((sample_features[f] - means_array[f]) / (sqrt(pow(stddevs_array[f], 2.0) + 1e-14)));

	return normalized_features;
}