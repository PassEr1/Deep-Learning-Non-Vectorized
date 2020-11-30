#include "math_utils.hpp"
#include <time.h>
#include <stdlib.h>
#include <random>
#include <vector>

double InitialValues::random(double from, double to)
{
	double n = (double)rand() / RAND_MAX ;
	return from + n * (to - from);
}

std::vector<double> InitialValues::random_vector(uint32_t size, double from, double to)
{
	std::vector<double> random_vec(size);
	random_vec.reserve(size);
	for (size_t i = 0; i < size; i++)
	{
		random_vec[i] = InitialValues::random(from, to);;
	}

	return std::move(random_vec);
}

double MathFunctions::sigmoid(double x)
{
	static const uint32_t ONE = 1.0;
	return  ONE / (ONE + exp(-x));
}
