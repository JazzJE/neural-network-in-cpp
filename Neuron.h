#pragma once
#include "StatisticsFunctions.h"
class Neuron
{
private:

	double* const neuron_weights;
	double* const neuron_bias;

	// for training with batch gradient descent
	double** const training_input_features;
	double** const training_activation_arrays;

	// for normal guessing and computation
	const int number_of_features;
	double* const input_features;
	double* const activation_array;
	double derived_value;

	// the neuron number is used to access the associated column of the activation arrays
	const int neuron_number;

	double* const running_mean;
	double* const running_variance;
	double* const scale;
	double* const shift;

	const double momentum;

public:

	// constructor
	Neuron(double* neuron_weights, double* neuron_bias, double* mean_and_variance, double* scale_and_shift,
		double** training_input_features, double** training_activation_arrays, double* input_features, double* activation_array,
		int number_of_features, int neuron_number);

	// function to compute the activation value and derived value
	double reLU_activation_function();
	void affinal_transform();

	// methods to compute training values


	// mutator/setter methods
	void set_derived_value(double d);

	// acessor/getter methods
	double get_derived_value();
};

