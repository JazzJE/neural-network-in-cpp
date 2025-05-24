#pragma once
#include <iostream>
#include <fstream>

char option_menu_choice();
void update_weights_and_biases_file(double*** weights, double** biases, std::fstream& weights_and_biases_file);