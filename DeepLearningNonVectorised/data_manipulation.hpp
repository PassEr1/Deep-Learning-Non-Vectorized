#pragma once
#include "data_loader.hpp"
#include <vector>
#include <functional>

namespace DataManipulation {
	void randomize(PDataCollection data_set);
	std::vector<double> math_operation_2_vectors(std::vector<double>& dest, const std::vector<double>& from, std::function<double(double first, double second)> _operator);
	std::vector<double> substract_vector(std::vector<double>& substracted, const std::vector<double>& substracting);
	std::vector<double> add_vector(std::vector<double>& added, const std::vector<double>& to_add);
};
