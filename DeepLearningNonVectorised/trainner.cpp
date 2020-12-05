#include "trainner.hpp"
#include "data_loader.hpp"
#include <iostream>

void SGD_Trainner::train(NetworkFullyConnected& network, double learning_rate, double threshold_stop_training, uint32_t batch_size, uint32_t epochs, PDataCollection data_set)
{
	std::wcout << network.predict(data_set->at(0))[0]<< L" "<< network.predict(data_set->at(0))[1] << std::endl;
}
