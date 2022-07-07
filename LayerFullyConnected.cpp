#include "LayerFullyConnected.h"

LayerFullyConnected::LayerFullyConnected(int prevNode_, int node_)
{
	nodeNum = node_;
	weight.initWeight(prevNode_, node_);
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
	nodes = actF->activation(y);
	return nodes;
}

Tensor<double> LayerFullyConnected::backwardProp(const Tensor<double>& e_in_)
{
	Tensor<double> dactNode = actF->deactivation(nodes);
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
	int dataNum = errors.dim(0);
	int prevNodeNum = prevNodes.dim(1);
	int postNodeNum = nodes.dim(1);
	Tensor<double> d_w(prevNodeNum, postNodeNum);
	double d_b = 0.f;
	for (int i = 0; i < prevNodeNum; i++)
	{
		for (int j = 0; j < postNodeNum; j++)
		{
			double sum = 0.f;
			for (int k = 0; k < dataNum; k++)
			{
				sum += errors.element(k, j) * prevNodes.element(k, i);
			}
			d_w(i, j) = 0.001 * sum / dataNum;
		}
	}
	for (int j = 0; j < postNodeNum; j++)
	{
		for (int k = 0; k < dataNum; k++)
		{
			d_b += 0.001 * errors.element(k, j) / (dataNum * postNodeNum);
		}
	}
	Tensor<double> w = weight.getWeight();
	double b = weight.getBias();
	w = w + d_w;
	b = b + d_b;
	weight.setWeight(w);
	weight.setBias(b);
}