#include "LayerFullyConnected.h"

LayerFullyConnected::LayerFullyConnected(OPT_INIT weightInit_, int prevNode_, int node_)
{
	nodeNum = node_;
	weight.initWeight(weightInit_, prevNode_, node_);
}

LayerFullyConnected::~LayerFullyConnected()
{

}

Tensor<double> LayerFullyConnected::forwardProp(const Tensor<double>& x_)
{
	Tensor<double> w = weight.getWeight();
	double b = weight.getBias();
	int dataNum = x_.dim(0);
	int prevNodeNum = w.dim(0);
	int postNodeNum = w.dim(1);
	Tensor<double> y(dataNum, postNodeNum);

	for (int i = 0; i < dataNum; i++)
	{
		for (int j = 0; j < postNodeNum; j++)
		{
			double sum = 0.f;
			for (int k = 0; k < prevNodeNum; k++)
			{
				sum += x_.element(i, k) * w.element(k, j);
			}
			y(i, j) = sum + b;
		}
	}
	nodes = actFnc->activation(y);
	return nodes;
}

Tensor<double> LayerFullyConnected::backwardProp(const Tensor<double>& e_in_)
{
	Tensor<double> dactNode = actFnc->deactivation(nodes);
	errors = dactNode * e_in_;

	Tensor<double> w = weight.getWeight();
	int dataNum = errors.dim(0);
	int prevNodeNum = w.dim(0);
	int postNodeNum = w.dim(1);
	Tensor<double> e_in(dataNum, prevNodeNum);
	for (int i = 0; i < dataNum; i++)
	{
		for (int j = 0; j < prevNodeNum; j++)
		{
			double sum = 0.f;
			for (int k = 0; k < postNodeNum; k++)
			{
				sum += errors.element(i, k) * w.element(j, k);
			}
			e_in(i, j) = sum;
		}
	}
	return e_in;
}

void LayerFullyConnected::updateWeight()
{
	Tensor<double> prevNodes = getPrevNodes();
	Tensor<double> w = weight.getWeight();
	double b = weight.getBias();
	Tensor<double> d_w = optimizer->optimizeDeltaWeight(errors, prevNodes);
	double d_b = optimizer->optimizeDeltaBias(errors);
	w = w + d_w;
	b = b + d_b;
	weight.setWeight(w);
	weight.setBias(b);
}