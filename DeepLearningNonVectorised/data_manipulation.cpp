#include "data_manipulation.hpp"
#include "exceptions.hpp"
#include <algorithm>


void DataManipulation::randomize(PDataCollection data_set)
{
	std::random_shuffle(data_set->begin(), data_set->end());
}

void DataManipulation::math_operation_2_vectors(std::vector<double>& dest, const std::vector<double>& from, std::function<double(double first, double second)> _operator)
{
	for (size_t i = 0; i < dest.size(); i++)
	{
		dest[i] = _operator(dest[i], from[i]);
	}
}

void DataManipulation::substract_vector(std::vector<double>& substracted, const std::vector<double>& substracting)
{
	if(substracted.size() != substracting.size())
	{
		throw MyException(ErrorCode::VECTOR_SUBSTRACTION_LENGTHS_ERROR);
	}

	DataManipulation::math_operation_2_vectors(substracted, substracting, [](double a, double b) {return a - b; });
}

void DataManipulation::add_vector(std::vector<double>& added, const std::vector<double>& to_add)
{
	if (added.size() != to_add.size())
	{
		throw MyException(ErrorCode::VECTOR_ADD_LENGTHS_ERROR);
	}


	DataManipulation::math_operation_2_vectors(added, to_add, [](double a, double b) {return a + b; });
}

