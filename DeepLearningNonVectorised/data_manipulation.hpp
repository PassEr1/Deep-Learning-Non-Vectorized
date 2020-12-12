#pragma once
#include "data_loader.hpp"
#include <vector>
#include <functional>

namespace DataManipulation {
	void randomize(PDataCollection data_set);
	void math_operation_2_vectors(std::vector<double>& dest, const std::vector<double>& from, std::function<double(double first, double second)> _operator);
	void substract_vector(std::vector<double>& substracted, const std::vector<double>& substracting);
	void add_vector(std::vector<double>& added, const std::vector<double>& to_add);
};
