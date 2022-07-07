#include "LayerInput.h"

LayerInput::LayerInput(int node_)
{
	nodeNum = node_;
}

LayerInput::~LayerInput()
{

}

Tensor<double> LayerInput::forwardProp(const Tensor<double>& x_)
{
	nodes = x_;
	return x_;
}

Tensor<double> LayerInput::backwardProp(const Tensor<double>& e_in_)
{
	return e_in_;
}

void LayerInput::updateWeight()
{

}