#include "trainner.hpp"
#include "data_loader.hpp"
#include "data_manipulation.hpp"
#include "exceptions.hpp"
#include <iostream>

void SGD_Trainner::train(NetworkFullyConnected& network, double learning_rate, double threshold_stop_training, uint32_t batch_size, uint32_t epochs, PDataCollection data_set)
{
	DataManipulation::randomize(data_set);
	static const uint32_t FIRST_DATA_SAMPLE = 0;
	static const uint32_t FIRST_LAYER_INDEX = 0;
	const uint32_t output_feature_count = data_set->at(FIRST_DATA_SAMPLE).size() - network.size_at(FIRST_LAYER_INDEX);

	for (size_t _ = 0; _ < epochs; _++)
	{
		std::vector<PDataCollection> batches = {data_set}; //batches_from_data_set();

		for(PDataCollection batch: batches)
		{
			double error_for_whole_batch = 0;
			network.reset_errors_and_derivatives();

			for (size_t sample_index = 0; sample_index < batch->size(); sample_index++)
			{
				std::vector<double> expected(
					batch->at(sample_index).end() - output_feature_count, 
					batch->at(sample_index).end());

				//forward propergation
				std::vector<double> output = network.predict(batch->at(sample_index));
				error_for_whole_batch += CostFunctions::logorithmic_cost_function__one_sample(output, expected);

				// compute overall cost (last layer)
				DataManipulation::substract_vector(output, expected); // TODO: return here an object and define a new variable

				// backpropergation all errors and derivatives
				network.backpropergation_error(output);
			}

			error_for_whole_batch = error_for_whole_batch / (-static_cast<signed long>(batch->size()));
			std::wcout << "Cost Function: " << error_for_whole_batch << std::endl;

			double factor = learning_rate / batch_size;
			network.update_weights(factor);
		}
	}

	std::wcout << "*** this is a test prediction: \n";
	auto prediction = network.predict(data_set->at(0));
	for (double input : data_set->at(0)) { std::wcout << input << " ";}
	std::wcout << std::endl;
	std::wcout << prediction[0]<< L" " << prediction[1] << std::endl;
}

double CostFunctions::logorithmic_cost_function__one_sample(const std::vector<double>& predicted, const std::vector<double>& expected)
{
	if (predicted.size() != expected.size())
	{
		throw MyException(ErrorCode::LOGORITHMIC_COST_FUNCTION_ONE_SAMPLE_PARAMETERS_LENGTH_ERROR);
	}
	
	double result = 0;
	for (size_t neuron_index = 0; neuron_index < predicted.size(); neuron_index++)
	{
		const double x = predicted[neuron_index];
		const double y = expected[neuron_index];	
		result += y * log(x) + (1 - y) * log(1- x);
	}

	return result;
}
