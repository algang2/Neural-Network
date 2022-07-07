#include "Layer.h"
#include "LayerInput.h"
#include "LayerFullyConnected.h"

#include "Function.h"

Layer::Layer() 
{
	nodeNum = 0;
	actF = nullptr;
}

Layer::~Layer() 
{
	delete prevLayer;
	delete actF;
}

Layer* Layer::initLayer(OPT_LYR layer_, int node_, Layer* prevLayer_)
{
	Layer* lyrptr = nullptr;
	if (layer_ == OPT_LYR::INPUT)
	{
		lyrptr = new LayerInput(node_);
	}
	else if (layer_ == OPT_LYR::FULLYCONNECTED)
	{
		int nodeNum = prevLayer_->nodeNum;
		lyrptr = new LayerFullyConnected(nodeNum, node_);
		lyrptr->prevLayer = prevLayer_;
	}
	return lyrptr;
}

void Layer::setActF(OPT_ACTF actF_)
{
	if (actF_ == OPT_ACTF::LINEAR)
	{
		actF = new Linear();
	}
	else if (actF_ == OPT_ACTF::SIGMOID)
	{
		actF = new Sigmoid();
	}
	else if (actF_ == OPT_ACTF::TANSIG)
	{
		actF = new Tansig();
	}
	else if (actF_ == OPT_ACTF::RELU)
	{
		actF = new ReLU();
	}
	else if (actF_ == OPT_ACTF::SOFTMAX)
	{
		actF = new Softmax();
	}
}

Tensor<double> Layer::getPrevNodes()
{
	return prevLayer->nodes;
}