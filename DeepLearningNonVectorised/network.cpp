#include "network.hpp"
#include "exceptions.hpp"
#include <time.h>
#include <random>
#include <memory>

NetworkFullyConnected::NetworkFullyConnected(std::vector<uint32_t> inputs_per_layer, ActivitionFunction default_activition)
:_layers(build_layers(inputs_per_layer, default_activition))
{}

std::vector<double> NetworkFullyConnected::predict(const std::vector<double>& inputs_to_feed)
{
	static const uint32_t INPUT_LAYER_INDEX = 0;
	const uint32_t neurons_to_feed = _layers[INPUT_LAYER_INDEX].size();
	for (size_t input_node_index = 0; input_node_index < neurons_to_feed; input_node_index ++)
	{
		double input_j = inputs_to_feed[input_node_index];
		std::static_pointer_cast<InputNeuron>(_layers[INPUT_LAYER_INDEX][input_node_index])->set_input(input_j);
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

void NetworkFullyConnected::zero_errors_and_derivative_for_layer(uint32_t layer_index)
{
	if (layer_index >= _layers.size())
	{
		throw MyException(ErrorCode::ZERO_DELTA_FOR_LAYER_INDEX_ERROR);
	}

	for (size_t node_index = 0; node_index < _layers[layer_index].size(); node_index++)
	{
		static const uint32_t RESET_VALUE = 0;
		std::static_pointer_cast<Neuron>(_layers[layer_index][node_index])->set_delta(RESET_VALUE);
		std::static_pointer_cast<Neuron>(_layers[layer_index][node_index])->reset_partial_derivative();
	}
}

void NetworkFullyConnected::compute_layer_partial_derivatives(uint32_t layer_index)
{
	if (layer_index >= _layers.size())
	{
		throw MyException(ErrorCode::COMPUTE_LAYER_PARTIAL_DERIVATIVES_INDEX_ERROR);
	}

	for (auto neuron_ptr : _layers[layer_index])
	{
		std::static_pointer_cast<Neuron>(neuron_ptr)->compute_partial_derivative();
	}
}

void NetworkFullyConnected::set_delta_of_layer(const std::vector<double>& deltas, uint32_t layer_index)
{
	if (_layers[layer_index].size() != deltas.size())
	{
		throw MyException(ErrorCode::SET_DELTA_OF_LAYER_VECTOR_SIZE_ERROR);
	}

	for (size_t node_index = 0; node_index < _layers[layer_index].size(); node_index++)
	{
		std::static_pointer_cast<Neuron>(_layers[layer_index][node_index])->set_delta(deltas[node_index]);
	}
}

uint32_t NetworkFullyConnected::layers_count() const
{
	return _layers.size();
}


NetworkFullyConnected::Layer NetworkFullyConnected::build_layer(
	uint32_t nodes_count, 
	ActivitionFunction activition, 
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
	std::for_each(layer.begin(), layer.end(),
		[&](Neuron::PtrNeuron& neuron) 
		{
			neuron->fire(); 
		});
}

std::vector<NetworkFullyConnected::Layer> NetworkFullyConnected::build_layers(const std::vector<uint32_t>& nodes_per_layer, ActivitionFunction activition)
{
	static const uint32_t MIN_LAYERS_COUNT_THRESHOLD = 3;
	if (nodes_per_layer.size() < MIN_LAYERS_COUNT_THRESHOLD)
	{
		throw MyException(ErrorCode::NETWORK_ARCHITECTURE_ERROR);
	}

	std::vector<NetworkFullyConnected::Layer> layers(nodes_per_layer.size());
	layers.reserve(nodes_per_layer.size());
	
	static const uint32_t INPUT_LAYER = 0;
	for (size_t input_node_index = 0; input_node_index < nodes_per_layer[INPUT_LAYER]; input_node_index++)
	{
		layers[INPUT_LAYER].push_back(std::make_shared<InputNeuron>());
	}

	static const std::function<void (void)> one_time_environment_random_helper = [](){ srand(time(NULL)); };
	one_time_environment_random_helper();

	static const uint32_t SKIP_INPUT_LAYER = 1;
	for (size_t layer_index = SKIP_INPUT_LAYER; layer_index < nodes_per_layer.size(); layer_index++)
	{
		static const uint32_t PREVIOUS_LAYER_INDEX = 1;
		layers[layer_index] = build_layer(nodes_per_layer[layer_index],
			activition,
			layers[layer_index - PREVIOUS_LAYER_INDEX]);
	}

	return std::move(layers);
}

void NetworkFullyConnected::backpropergation_delta_single_layer(uint32_t from_layer)
{
	for (Neuron::PtrNeuron neuron: _layers[from_layer])
	{
		std::static_pointer_cast<Neuron>(neuron)-> backpropergate_delta();
	}
}


void NetworkFullyConnected::backpropergation_error(const std::vector<double>& output_layer_error)
{
	static const uint32_t TO_INDEX = 1;
	uint32_t last_layer_index = layers_count() - TO_INDEX;
	set_delta_of_layer(output_layer_error, last_layer_index);

	static const uint32_t LAYER_TO_STOP_PROPERGATION = 1;

	for (size_t layer_index = last_layer_index;
		layer_index > LAYER_TO_STOP_PROPERGATION;
		layer_index--)
	{

		backpropergation_delta_single_layer(layer_index);
		compute_layer_partial_derivatives(layer_index);
	}

	compute_layer_partial_derivatives(LAYER_TO_STOP_PROPERGATION);
}

void NetworkFullyConnected::reset_errors_and_derivatives()
{
	static const uint32_t START_LINE = 1;
	for (size_t i = START_LINE; i < layers_count(); i++)
	{
		zero_errors_and_derivative_for_layer(i);
	}
}

void NetworkFullyConnected::update_weights(double factor)
{
	static const uint32_t START_LAYER = 1;
	for (size_t layer_index = START_LAYER; layer_index < layers_count(); layer_index++)
	{
		for (auto& neuron: _layers[layer_index])
		{
			std::static_pointer_cast<Neuron>(neuron)->update_weights(factor);
		}
	}
}

uint32_t NetworkFullyConnected::size_at(uint32_t layer_index) const 
{
	return _layers[layer_index].size();
}


