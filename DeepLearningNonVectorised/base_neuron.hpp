#pragma once

class BaseNeuron {
public:
	virtual void fire()=0;
	virtual double get_output() const=0;
};