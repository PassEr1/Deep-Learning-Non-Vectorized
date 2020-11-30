#pragma once
#include "base_neuron.hpp"


class InputNeuron final
:public BaseNeuron
{
public:
	InputNeuron(double value);

public:
	InputNeuron(const InputNeuron&) = delete;
	InputNeuron(InputNeuron&&) = delete;
	InputNeuron& operator=(const InputNeuron&) = delete;
	InputNeuron& operator=(InputNeuron&&) = delete;

public:
	double get_output() const override;
	void fire() override;

private:
	const double _value;

};