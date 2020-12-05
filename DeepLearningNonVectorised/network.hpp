#pragma once
#include "neuron.hpp"
#include "input_neuron.hpp"
#include "data_loader.hpp"
#include <cstdint>
#include <memory>

class NetworkFullyConnected final
{
	using Layer = std::vector<Neuron::PtrNeuron>;

public:
	NetworkFullyConnected(std::vector<uint32_t> inputs_per_layer);

public:
	std::vector<double> predict(const std::vector<double>& inputs_to_feed);

public:
	NetworkFullyConnected(const NetworkFullyConnected&) = delete;
	NetworkFullyConnected(NetworkFullyConnected&&) = delete;
	NetworkFullyConnected& operator=(const NetworkFullyConnected&) = delete;
	NetworkFullyConnected& operator=(NetworkFullyConnected&&) = delete;

private:
	static std::vector<NetworkFullyConnected::Layer> build_layers(std::vector<uint32_t> inputs_per_layer);
	static Layer build_layer(uint32_t len, activition_signature activition, 
		std::vector<Neuron::PtrNeuron> input_connection);
	static void layer_fire(Layer& layer);

private:
	void forward_propergation();

private:
	std::vector<Layer> _layers;
};
