#include "MenuFunctions.h"
char option_menu_choice()
{
	char option;
	std::cout << "\nOption Menu:"
		<< "\n\t1. Train neural network (five-fold, batch gradient descent)"
		<< "\n\t2. Predict a value"
		<< "\n\t3. Update your neural network to latest trained version"
		<< "\n\t4. Exit program"
		<< "\nPlease select an option: ";
	std::cin >> option;

	while (option < '1' || option > '4')
	{
		std::cout << "[ERROR] Please enter a valid input (1-4): ";
		std::cin >> option;
	}

	return option;
}

// update the weights and biases
void update_weights_and_biases_file(double*** weights, double** biases, std::string weights_and_biases_file_name, 
	int* number_of_neurons_each_hidden_layer, int number_of_hidden_layers)
{

}