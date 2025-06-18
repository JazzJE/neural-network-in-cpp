#pragma once
#include "StatisticsFunctions.h"
class Neuron
{
private:

	double* const neuron_weights;
	double* const neuron_bias;

	// for normal guessing and computation
	const int number_of_features;
	double* const input_features;
	double* const activation_array;

	// for training with batch gradient descent
	double** const training_input_features;
	double** const training_activation_arrays;
	double* const derived_values;
	double average_derived_value;
	double training_mean;
	double training_variance;

	const double momentum;
	const int batch_size;

	// the neuron number is used to access the associated column of the activation arrays
	const int neuron_number;

	double* const running_mean;
	double* const running_variance;
	double* const scale;
	double* const shift;

public:

	// constructor
	Neuron(double* neuron_weights, double* neuron_bias, double* mean_and_variance, double* scale_and_shift,
		double** training_input_features, double** training_activation_arrays, double* input_features, double* activation_array,
		int number_of_features, int batch_size, int neuron_number);

	// delete the dynamic memory still left, which is the derived values
	~Neuron();

	// methods for calculating activation values for normal input features
	virtual void compute_activation_value();
	void linear_transform();
	void normalize_activation_value();
	void affinal_transform();
	void relu_activation_function();

	// methods to compute values for training
	virtual void training_compute_activation_values();
	void training_linear_transform();
	void training_normalize_activation_value();
	void training_affinal_transform();
	void training_relu_activation_function();

	// acessor/getter methods
	double* get_derived_values() const;
	double get_training_mean() const;
	double get_training_variance() const;
};

