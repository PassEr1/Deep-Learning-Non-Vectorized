#include "neuron.hpp"
#include "exceptions.hpp"
#include "math_consts.hpp"
#include <algorithm>

Neuron::Neuron(activition_signature activition_func, std::vector<BaseNeuron*> connections)
	:_activition(activition_func),
	_connections(connections),
	_delta(0),
	_derivative(0),
	_bias(InitialValues::random(-MathConsts::EPSILON, MathConsts::EPSILON)),
	_output(0),
	_weights(InitialValues::random_vector(connections.size(), -MathConsts::EPSILON, MathConsts::EPSILON)),
	_inputs(connections.size(), 0)
{}

void Neuron::fire()
{
	read_inputs();
	double z = 0;
	for (uint32_t input_index = 0; input_index < _inputs.size(); input_index++)
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

	for (size_t wi = 0; wi < _weights.size(); wi++)
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
	for(BaseNeuron* connection: _connections)
	{
		_inputs[index_to_save_input] = connection->get_output();
		index_to_save_input++;
	}
}
