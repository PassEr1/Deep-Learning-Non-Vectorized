#include "data_manipulation.hpp"
#include "exceptions.hpp"
#include <algorithm>


void DataManipulation::randomize(PDataCollection data_set)
{
	std::random_shuffle(data_set->begin(), data_set->end());
}

std::vector<double> DataManipulation::math_operation_2_vectors(std::vector<double>& first, const std::vector<double>& second, std::function<double(double first, double second)> _operator)
{
	std::vector<double> result(first.size());
	result.reserve(first.size());

	for (size_t i = 0; i < first.size(); i++)
	{
		result[i] = _operator(first[i], second[i]);
	}

	return std::move(result);
}

std::vector<double> DataManipulation::substract_vector(std::vector<double>& substracted, const std::vector<double>& substracting)
{
	if(substracted.size() != substracting.size())
	{
		throw MyException(ErrorCode::VECTOR_SUBSTRACTION_LENGTHS_ERROR);
	}

	return DataManipulation::math_operation_2_vectors(substracted, substracting, [](double a, double b) {return a - b; });
}

std::vector<double> DataManipulation::add_vector(std::vector<double>& added, const std::vector<double>& to_add)
{
	if (added.size() != to_add.size())
	{
		throw MyException(ErrorCode::VECTOR_ADD_LENGTHS_ERROR);
	}


	return DataManipulation::math_operation_2_vectors(added, to_add, [](double a, double b) {return a + b; });
}

