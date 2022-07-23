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
	static Layer* initLayer(OPT_LYR layer_, int depth_, int kernel_, int stride_, OPT_INIT weightInit_, Layer* prevLayer_);
	void setActFunction(OPT_ACTF actFnc_);
	void setOptimizer(OPT_OPTM optimizer_, const double& learningRate_, const double& val_0_ = 0.f, const double& val_1_ = 0.f);
	Tensor<double> getPrevNodes();
	int getDim(int dim_);
	int getPrevDim(int dim_);
	virtual Tensor<double> forwardProp(const Tensor<double>& x_) = 0;
	virtual Tensor<double> backwardProp(const Tensor<double>& e_in_) = 0;
	virtual void updateWeight() = 0;

	virtual std::string saveLayer() = 0;
	static Layer* loadLayer(BinaryReader& reader_, Layer* prevLayer_);
protected:
	OPT_LYR type;
	int *dim;
	OPT_ACTF actF;
	Layer* prevLayer;
	ActFunction* actFnc;
	Optimizer* optimizer;
	Tensor<double> nodes;
	Tensor<double> errors;
	Weight weight;
};