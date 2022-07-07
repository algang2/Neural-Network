#pragma once
#include "Option.h"
#include "Function.h"
#include "Weight.h"

class Layer
{
public:
	Layer();
	~Layer();
	static Layer* initLayer(OPT_LYR layer_, int node_, Layer* prevLayer_ = nullptr);
	void setActF(OPT_ACTF actF_);
	Tensor<double> getPrevNodes();
	virtual Tensor<double> forwardProp(const Tensor<double>& x_) = 0;
	virtual Tensor<double> backwardProp(const Tensor<double>& e_in_) = 0;
	virtual void updateWeight() = 0;
protected:
	int nodeNum;
	Layer* prevLayer;
	ActFunction* actF;
	Tensor<double> nodes;
	Tensor<double> errors;
	Weight weight;
};