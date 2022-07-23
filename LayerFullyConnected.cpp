#include "LayerFullyConnected.h"

LayerFullyConnected::LayerFullyConnected(OPT_INIT weightInit_, int prevNode_, int node_)
{
	dim[0] = node_;
	dim[1] = 1;
	dim[2] = 1;
	type = OPT_LYR::FULLYCONNECTED;
	weight.initWeight(weightInit_, prevNode_, node_);
}

LayerFullyConnected::LayerFullyConnected(BinaryReader& reader_)
{
	type = OPT_LYR::FULLYCONNECTED;
	dim[0] = stoi(reader_.getNext());
	dim[1] = stoi(reader_.getNext());
	dim[2] = stoi(reader_.getNext());
	actF = (OPT_ACTF)stoi(reader_.getNext());
	setActFunction(actF);
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

std::string LayerFullyConnected::saveLayer()
{
	BinaryReader reader;
	reader.setNext(std::to_string((int)type), true);
	reader.setNext(std::to_string(dim[0]));
	reader.setNext(std::to_string(dim[1]));
	reader.setNext(std::to_string(dim[2]));
	reader.setNext(std::to_string((int)actF));
	return reader.getBinary();
}