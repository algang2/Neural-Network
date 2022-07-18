#pragma once
#include "Option.h"
#include "ActiveFunction.h"
#include "Weight.h"
#include "Optimizer.h"
#include "NeuralNet.h"

class Layer
{
	friend class NeuralNet;
public:
	Layer();
	~Layer();
	static Layer* initLayer(OPT_LYR layer_, int node_, OPT_INIT weightInit_, Layer* prevLayer_ = nullptr);
	void setActFunction(OPT_ACTF actFnc_);
	void setOptimizer(OPT_OPTM optimizer_, const double& learningRate_, const double& val_0_ = 0.f, const double& val_1_ = 0.f);
	Tensor<double> getPrevNodes();
	virtual Tensor<double> forwardProp(const Tensor<double>& x_) = 0;
	virtual Tensor<double> backwardProp(const Tensor<double>& e_in_) = 0;
	virtual void updateWeight() = 0;
protected:
	OPT_LYR type;
	int nodeNum;
	OPT_ACTF actF;
	Layer* prevLayer;
	ActFunction* actFnc;
	Optimizer* optimizer;
	Tensor<double> nodes;
	Tensor<double> errors;
	Weight weight;
};