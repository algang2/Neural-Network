#pragma once
#include "Layer.h"

class LayerFullyConnected :public Layer
{
public:
	LayerFullyConnected(int prevNode_, int node_);
	~LayerFullyConnected();
	virtual Tensor<double> forwardProp(const Tensor<double>& x_) override;
	virtual Tensor<double> backwardProp(const Tensor<double>& e_in_) override;
	virtual void updateWeight() override;
};