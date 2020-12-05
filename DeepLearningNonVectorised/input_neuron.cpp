#include "input_neuron.hpp"


double InputNeuron::get_output() const
{
	return _value;
}

void InputNeuron::fire()
{}

void InputNeuron::set_input(double value)
{
	_value = value;
}
