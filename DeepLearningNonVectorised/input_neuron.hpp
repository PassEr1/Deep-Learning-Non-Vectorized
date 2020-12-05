#pragma once
#include "base_neuron.hpp"


class InputNeuron final
:public BaseNeuron
{
public:
	InputNeuron() = default;
	InputNeuron(const InputNeuron&) = delete;
	InputNeuron(InputNeuron&&) = delete;
	InputNeuron& operator=(const InputNeuron&) = delete;
	InputNeuron& operator=(InputNeuron&&) = delete;

public:
	double get_output() const override;
	void fire() override;
	void set_input(double value);

private:
	 double _value;

};