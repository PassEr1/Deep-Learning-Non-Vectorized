#include "neuron.hpp"
#include "exceptions.hpp"
#include "math_consts.hpp"
#include <algorithm>
static const double ZERO_INITIATION = 0;

Neuron::Neuron(activition_signature activition_func, std::vector<Neuron::PtrNeuron> connections)
	:_activition(activition_func),
	_connections(connections),
	_delta(ZERO_INITIATION),
	_derivative(ZERO_INITIATION),
	_bias(InitialValues::random(-MathConsts::EPSILON, MathConsts::EPSILON)),
	_output(ZERO_INITIATION),
	_weights(InitialValues::random_vector(connections.size(), -MathConsts::EPSILON, MathConsts::EPSILON)),
	_inputs(connections.size(), ZERO_INITIATION)
{}

void Neuron::fire()
{
	read_inputs();
	double z = ZERO_INITIATION;
	static const uint32_t FIRST_NODE_INDEX = 0;
	for (uint32_t input_index = FIRST_NODE_INDEX; input_index < _inputs.size(); input_index++)
	{
		z += _weights[input_index] * _inputs[input_index];
	}

	z += _bias;
	_output = _activition(z);
}

void Neuron::add_to_bias(double add_to_bias)
{
	_bias += add_to_bias;
}

void Neuron::add_to_weights(std::vector<double> add_to_weights)
{
	if (add_to_weights.size() != _weights.size())
	{
		throw MyException(ErrorCode::WEIGHTS_FORM_NOT_MATH);
	}

	static const uint32_t FIRST_WEIGHT_INDEX = 0;
	for (size_t wi = FIRST_WEIGHT_INDEX; wi < _weights.size(); wi++)
	{
		_weights[wi] += add_to_weights[wi];
	}
}

double Neuron::get_bias() const
{
	return _bias;
}

double Neuron::get_delta() const
{
	return _delta;
}

double Neuron::get_output() const
{
	return _output;
}

void Neuron::read_inputs()
{
	uint32_t index_to_save_input=0;
	for (size_t connection_index = 0; connection_index < _connections.size(); connection_index++)
	{
		_inputs[index_to_save_input] = _connections[connection_index]->get_output();
		index_to_save_input++;
	}
}
