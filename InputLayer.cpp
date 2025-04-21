#include "InputLayer.h"

// no weights in the input layer and the number of neurons is only the number of features
InputLayer::InputLayer(int number_of_features) : Layer(number_of_features, 0)
{ }

double* InputLayer::output_activation_array()
{
	return input_features;
}