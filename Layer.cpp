#include "Layer.h"

// constructor to initialize neurons 
Layer::Layer(int number_of_neurons_in_layer, int number_of_neurons_in_prior_layer)
{
	// initialize private variables
	this->number_of_neurons_in_layer = number_of_neurons_in_layer;
	this->number_of_neurons_in_prior_layer = number_of_neurons_in_prior_layer;

	
}