#pragma once
#include <functional>
#include <vector>
#include <cstdint>


using ActivitionFunction = std::function<double(double)>;

namespace InitialValues {
	double random(double from, double to);
	std::vector<double> random_vector(uint32_t size, double from, double to);
}

namespace MathFunctions
{
	double sigmoid(double x);
}

