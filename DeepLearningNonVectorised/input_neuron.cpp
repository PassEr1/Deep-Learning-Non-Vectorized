#include "input_neuron.hpp"

InputNeuron::InputNeuron(double value)
:_value(value)
{}

double InputNeuron::get_output() const
{
	return _value;
}

void InputNeuron::fire()
{}
