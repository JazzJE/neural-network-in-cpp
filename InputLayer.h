
#include "Layer.h"
class InputLayer : public Layer
{
public:

	// pass in the number of neurons in the input as the number of features; passing into superclass can be seen in implementation file
	InputLayer(int number_of_features);

	// for an input layer, the activations are simply the features provided without any changes
	double* output_activation_array() override;
};
