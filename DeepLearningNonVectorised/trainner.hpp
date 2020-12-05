#pragma once
#include "network.hpp"

namespace SGD_Trainner {
	void train(NetworkFullyConnected& network, double learning_rate, double threshold_stop_training, uint32_t batch_size, uint32_t epochs, PDataCollection data_set);
};

