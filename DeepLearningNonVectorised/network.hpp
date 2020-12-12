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
	NetworkFullyConnected(std::vector<uint32_t> inputs_per_layer, ActivitionFunction default_activition);

public:
	std::vector<double> predict(const std::vector<double>& inputs_to_feed);
	void backpropergation_error(const std::vector<double>& output_layer_error);
	void reset_errors_and_derivatives();
	void update_weights(double factor);

public:
	uint32_t layers_count() const;
	uint32_t size_at(uint32_t layer_index) const;

	
public:
	NetworkFullyConnected(const NetworkFullyConnected&) = delete;
	NetworkFullyConnected(NetworkFullyConnected&&) = delete;
	NetworkFullyConnected& operator=(const NetworkFullyConnected&) = delete;
	NetworkFullyConnected& operator=(NetworkFullyConnected&&) = delete;

private:
	static Layer build_layer(uint32_t len, ActivitionFunction activition, 
		std::vector<Neuron::PtrNeuron> input_connection);
	static void layer_fire(Layer& layer);
	static std::vector<Layer> build_layers(const std::vector<uint32_t>& inputs_per_layer, ActivitionFunction activition);

private:
	void forward_propergation();
	void backpropergation_delta_single_layer(uint32_t from_layer);
	void set_delta_of_layer(const std::vector<double>& deltas, uint32_t layer_index);
	void zero_errors_and_derivative_for_layer(uint32_t layer_index);
	void compute_layer_partial_derivatives(uint32_t layer_index);


private:
	std::vector<Layer> _layers;
};
