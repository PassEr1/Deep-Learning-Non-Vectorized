#pragma once
#include "network.hpp"

namespace SGD_Trainner {
	void train(NetworkFullyConnected& network, double learning_rate, double threshold_stop_training, uint32_t batch_size, uint32_t epochs, PDataCollection data_set);
};

namespace CostFunctions {
	// to use this one for a total bunch of training batch, 
	//accumulate all results for each sample in it and multiply by (-1)/(batch size)
	double logorithmic_cost_function__one_sample(const std::vector<double>& predicted, const std::vector<double>& expected);
}

