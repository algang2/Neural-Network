#pragma once
#include "Layer.h"

class LayerInput :public Layer
{
public:
	LayerInput(int node_);
	~LayerInput();
	virtual Tensor<double> forwardProp(const Tensor<double>& x_) override;
	virtual Tensor<double> backwardProp(const Tensor<double>& e_in_) override;
	virtual void updateWeight() override;
};