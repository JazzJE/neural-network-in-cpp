#include "Neuron.h"
Neuron::Neuron(double* neuron_weights, double* neuron_bias, double* mean_and_variance, double* scale_and_shift,
	double** training_input_features, double** training_activation_arrays, double* input_features, double* activation_array,
	int number_of_features, int batch_size, int neuron_number) :
	neuron_weights(neuron_weights), neuron_bias(neuron_bias), running_mean(&mean_and_variance[0]), running_variance(&mean_and_variance[1]),
	scale(&scale_and_shift[0]), shift(&scale_and_shift[1]), number_of_features(number_of_features), momentum(0.9), 
	neuron_number(neuron_number), batch_size(batch_size), derived_values(new double[batch_size]),
	
	// link to the n-1th layer and the n+1th layer
	input_features(input_features), activation_array(activation_array),
	training_input_features(training_input_features), training_activation_arrays(training_activation_arrays)

{
	training_mean = 0;
	training_variance = 0;
	average_derived_value = 0;
}

Neuron::~Neuron()
{
	delete[] derived_values;
}

// methods for normal computation and prediction
// compute activation value of neuron for normal predictions
void Neuron::compute_activation_value()
{
	// calculate activation value
	linear_transform();

	// normalize output according to running mean and running variance
	normalize_activation_value();

	// affinal transform implementation
	affinal_transform();

	// relu function implementation
	relu_activation_function();
}

// calculate the activation value of the linear transform
void Neuron::linear_transform()
{
	activation_array[neuron_number] = 0;
	for (int f = 0; f < number_of_features; f++)
		activation_array[neuron_number] += neuron_weights[f] * input_features[f];

	activation_array[neuron_number] += *neuron_bias;
}

// normalize the activation value according to the running means and running variance
void Neuron::normalize_activation_value()
{ activation_array[neuron_number] = (activation_array[neuron_number] - (*running_mean)) / (sqrt(pow(*running_variance, 2.0) + 1e-14)); }

// scale the normalized activation value using the scale and shift
void Neuron::affinal_transform()
{ activation_array[neuron_number] = (*scale) * activation_array[neuron_number] + (*shift); }

// apply a relu activation function implementation to the output; if negative number, just make 0
void Neuron::relu_activation_function()
{ if (activation_array[neuron_number] <= 0) activation_array[neuron_number] = 0; }


// methods for training
void Neuron::training_compute_activation_values()
{
	// calculate the activation values of each linear transform of each sample in the batch
	training_linear_transform();

	// normalize each activation value in the batch
	training_normalize_activation_value();

	// transform each sample's activation value according to the current values of the scale and shift
	training_affinal_transform();

	// ensure value is above 0, else make it 0
	training_relu_activation_function();
}

// for each sample, calculate the activation value
void Neuron::training_linear_transform()
{
	for (int s = 0; s < batch_size; s++)
	{
		training_activation_arrays[s][neuron_number] = 0;
		for (int f = 0; f < number_of_features; f++)
			training_activation_arrays[s][neuron_number] += (training_input_features[s][f] * neuron_weights[f]);

		training_activation_arrays[s][neuron_number] += (*neuron_bias);
	}
}

// normalize each output activation value
void Neuron::training_normalize_activation_value()
{
	training_mean = 0;
	training_variance = 0;

	// calculate mean of the activation values
	for (int s = 0; s < batch_size; s++)
		training_mean += training_activation_arrays[s][neuron_number];
	training_mean /= batch_size;

	// calculate standard deviation of activation values
	for (int s = 0; s < batch_size; s++)
		training_variance += pow(training_activation_arrays[s][neuron_number] - training_mean, 2);
	training_variance /= batch_size;
	training_variance = sqrt(training_variance);

	// calculate all the normalized activation values using the aforementioned
	for (int s = 0; s < batch_size; s++)
		training_activation_arrays[s][neuron_number] = (training_activation_arrays[s][neuron_number] - training_mean) / (sqrt(pow(training_variance, 2) + 1e-14));
}

// transform the normalized values; must calculate the mean and variance at this step
void Neuron::training_affinal_transform()
{
	for (int s = 0; s < batch_size; s++)
		training_activation_arrays[s][neuron_number] = (*scale) * training_activation_arrays[s][neuron_number] + (*shift);
}

// go through all the normalized activations and just set them to 0 if less than or equal to 0
void Neuron::training_relu_activation_function()
{
	for (int s = 0; s < batch_size; s++)
		if (training_activation_arrays[s][neuron_number] <= 0) training_activation_arrays[s][neuron_number] = 0;
}

double* Neuron::get_derived_values() const
{ return derived_values; }

double Neuron::get_training_mean() const
{ return training_mean;  }

double Neuron::get_training_variance() const 
{ return training_variance;  }