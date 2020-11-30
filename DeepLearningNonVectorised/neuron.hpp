#pragma once
#include "math_utils.hpp"
#include "base_neuron.hpp"
#include <cstdint>
#include <string>
#include <vector>

class Neuron final
	:public BaseNeuron
{
public: 
	Neuron(activition_signature activition_func, std::vector<BaseNeuron*> connections);

public:
	void add_to_bias(double add_to_bias);
	void add_to_weights(std::vector<double> add_to_weights);

public:
	void fire() override;
	double get_output() const override;

public:
	double get_bias() const;
	double get_delta() const;


public:
	Neuron(const Neuron&) = delete;
	Neuron(Neuron&&) = delete;
	Neuron& operator=(const Neuron&) = delete;
	Neuron& operator=(Neuron&&) = delete;

private:
	void read_inputs();

private:
	const activition_signature _activition;
	const std::vector<BaseNeuron*> _connections;
	std::vector<double> _inputs;
	std::vector<double> _weights;
	double _delta;
	double _derivative;
	double _bias;
	double _output;
};
