#include "trainner.hpp"
#include "data_loader.hpp"
#include <iostream>

void SGD_Trainner::train(NetworkFullyConnected& network, double learning_rate, double threshold_stop_training, uint32_t batch_size, uint32_t epochs, PDataCollection data_set)
{
	for (size_t _ = 0; _ < epochs; _++)
	{
		//auto batches = batches_from_data_set();
		//for each batch
		{
			//for each_data_sample
			{
				//forward propergation
				// compute overall cost (last layer)
				// backpropergation all Delta
				//accumulate derivatives for all nodes(input * delta) and detlas only for bias nodes.		
			}

			//update weights
		}
	}
	std::wcout << network.predict(data_set->at(0))[0]<< L" "<< network.predict(data_set->at(0))[1] << std::endl;
}
