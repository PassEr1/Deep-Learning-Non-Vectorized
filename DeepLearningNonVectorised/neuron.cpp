#include "neuron.hpp"
#include "exceptions.hpp"
#include "math_consts.hpp"
#include "data_manipulation.hpp"
#include<iostream>
#include <algorithm>
static const double ZERO_INITIATION = 0;

Neuron::Neuron(ActivitionFunction activition_func, std::vector<Neuron::PtrNeuron> connections)
	:_activition(activition_func),
	_connections(connections),
	_delta(ZERO_INITIATION),
	_bias(InitialValues::random(-MathConsts::EPSILON, MathConsts::EPSILON)),
	_output(ZERO_INITIATION),
	_weights(InitialValues::random_vector(connections.size(), -MathConsts::EPSILON, MathConsts::EPSILON)),
	_inputs(connections.size(), ZERO_INITIATION),
	_partial_derivatives(connections.size(), ZERO_INITIATION)
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

void Neuron::add_to_delta(double add_to_delta)
{
	_delta += add_to_delta;
}

void Neuron::set_delta(double delta)
{
	_delta = delta;
}

void Neuron::backpropergate_delta()
{
	for (size_t neuron_index = 0; neuron_index < _connections.size(); neuron_index++)
	{
		const double partial_delta_for_input = _delta * _weights[neuron_index];
		std::static_pointer_cast<Neuron>(_connections[neuron_index])->add_to_delta(partial_delta_for_input);
	}
}

void Neuron::compute_partial_derivative()
{
	std::vector<double> new_partial_derivatives(_inputs);
	std::transform(new_partial_derivatives.begin(),
		new_partial_derivatives.end(),
		new_partial_derivatives.begin(),
		[&](auto one_input) {return one_input * _delta; });
	_partial_derivatives = DataManipulation::add_vector(_partial_derivatives, new_partial_derivatives);
	
}

void Neuron::reset_partial_derivative()
{
	static const uint32_t RESET_VALUE = 0;
	std::fill(_partial_derivatives.begin(), _partial_derivatives.end(), RESET_VALUE);
}

void Neuron::update_weights(double factor)
{
	_bias -= _delta * factor;
	std::vector<double> steps(_partial_derivatives);
	std::transform(steps.begin(), steps.end(), steps.begin(), [&factor](auto step_for_weight) {return step_for_weight * factor; });
	DataManipulation::substract_vector(_weights, steps);
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
