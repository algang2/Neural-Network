#include "Layer.h"
#include "LayerInput.h"
#include "LayerFullyConnected.h"

Layer::Layer() 
{
	nodeNum = 0;
	prevLayer = nullptr;
	actFnc = nullptr;
	optimizer = nullptr;
}

Layer::~Layer() 
{
	delete prevLayer;
	delete actFnc;
	delete optimizer;
}

Layer* Layer::initLayer(OPT_LYR layer_, int node_, OPT_INIT weightInit_, Layer* prevLayer_)
{
	Layer* lyrptr = nullptr;
	if (layer_ == OPT_LYR::INPUT)
	{
		lyrptr = new LayerInput(node_);
	}
	else if (layer_ == OPT_LYR::FULLYCONNECTED)
	{
		int nodeNum = prevLayer_->nodeNum;
		lyrptr = new LayerFullyConnected(weightInit_, nodeNum, node_);
		lyrptr->prevLayer = prevLayer_;
	}
	return lyrptr;
}

void Layer::setActFunction(OPT_ACTF actFnc_)
{
	actF = actFnc_;
	if (actFnc_ == OPT_ACTF::LINEAR)
	{
		actFnc = new Linear();
	}
	else if (actFnc_ == OPT_ACTF::SIGMOID)
	{
		actFnc = new Sigmoid();
	}
	else if (actFnc_ == OPT_ACTF::TANSIG)
	{
		actFnc = new Tansig();
	}
	else if (actFnc_ == OPT_ACTF::RELU)
	{
		actFnc = new ReLU();
	}
	else if (actFnc_ == OPT_ACTF::LEAKYRELU)
	{
		actFnc = new LeakyReLU();
	}
	else if (actFnc_ == OPT_ACTF::SOFTMAX)
	{
		actFnc = new Softmax();
	}
}

void Layer::setOptimizer(OPT_OPTM optimizer_, const double& learningRate_, const double& val_0_, const double& val_1_)
{
	Tensor<double> w = weight.getWeight();
	if (optimizer_ == OPT_OPTM::GDM)
	{
		double momentum = val_0_;
		if (optimizer != nullptr)
		{
			delete optimizer;
		}
		optimizer = new GDM(learningRate_, w, momentum);
	}
	else if (optimizer_ == OPT_OPTM::RMSPROP)
	{
		double gamma = val_0_;
		if (optimizer != nullptr)
		{
			delete optimizer;
		}
		optimizer = new RMSProp(learningRate_, w, gamma);
	}
	else if (optimizer_ == OPT_OPTM::ADAGRAD)
	{
		if (optimizer != nullptr)
		{
			delete optimizer;
		}
		optimizer = new Adagrad(learningRate_, w);
	}
	else if (optimizer_ == OPT_OPTM::ADAM)
	{
		double beta_1 = val_0_;
		double beta_2 = val_1_;
		if (optimizer != nullptr)
		{
			delete optimizer;
		}
		optimizer = new Adam(learningRate_, w, beta_1, beta_2);
	}
	else
	{
		if (optimizer != nullptr)
		{
			delete optimizer;
		}
		optimizer = new Null(learningRate_);
	}
}

Tensor<double> Layer::getPrevNodes()
{
	return prevLayer->nodes;
}