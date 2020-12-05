#include "network.hpp"
#include <time.h>
#include <random>

NetworkFullyConnected::NetworkFullyConnected(std::vector<uint32_t> inputs_per_layer)
:_layers()
{
	srand(time(NULL));
}

std::vector<double> NetworkFullyConnected::predict(const std::vector<double>& inputs_to_feed)
{
	static const uint32_t INPUT_LAYER_INDEX = 0;
	const uint32_t neurons_to_feed = _layers[INPUT_LAYER_INDEX].size();
	for (size_t input_node_index = 0; input_node_index < neurons_to_feed; input_node_index ++)
	{
		double input_j = inputs_to_feed[input_node_index];
		std::dynamic_pointer_cast<InputNeuron>(_layers[INPUT_LAYER_INDEX][input_node_index])->set_input(input_j);
	}

	forward_propergation();

	static const uint32_t TO_INDEX = 1;
	const uint32_t output_layer_index = _layers.size() - TO_INDEX;

	std::vector<double> predicated_output;
	predicated_output.reserve(_layers[output_layer_index].size());

	std::for_each(_layers[output_layer_index].begin(), _layers[output_layer_index].end(),
		[&predicated_output](Neuron::PtrNeuron& out_neuron)
 {
			predicated_output.push_back(out_neuron->get_output());
		});

	return predicated_output;
}

std::vector<NetworkFullyConnected::Layer> NetworkFullyConnected::build_layers(std::vector<uint32_t> inputs_per_layer)
{
	std::vector<NetworkFullyConnected::Layer> layers(inputs_per_layer.size());
	layers.reserve(inputs_per_layer.size());
	return std::move(layers);
}

NetworkFullyConnected::Layer NetworkFullyConnected::build_layer(
	uint32_t nodes_count, 
	activition_signature activition, 
	std::vector<Neuron::PtrNeuron> input_connection)
{
	Layer layer(nodes_count);
	layer.reserve(nodes_count);
	for (size_t node_index = 0; node_index < nodes_count; node_index++)
	{
		layer[node_index] = Neuron::PtrNeuron(new Neuron(activition, input_connection));
	}

	return std::move(layer);
}

void NetworkFullyConnected::forward_propergation()
{
	static const uint32_t FIRST_LAYER_TO_FIRE = 0;
	static const uint32_t TO_INDEX = 1;
	const uint32_t last_layer_to_fire = _layers.size() - TO_INDEX;

	for (size_t layer_index = FIRST_LAYER_TO_FIRE; layer_index <= last_layer_to_fire ; layer_index++)
	{
		NetworkFullyConnected::layer_fire(_layers[layer_index]);
	}
}

void NetworkFullyConnected::layer_fire(Layer& layer)
{
	std::for_each(layer.begin(), layer.end(), [&](Neuron::PtrNeuron& neuron) {neuron->fire(); });
}
