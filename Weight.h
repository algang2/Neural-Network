#pragma once
#include <random>
#include "Option.h"
#include "Tensor.h"

class Weight
{
public:
	Weight();
	~Weight();
	void initWeight(OPT_INIT weightInit_, int dim_0_, int dim_1_);
	void initWeight(OPT_INIT weightInit_, int dim_0_, int dim_1_, int dim_2_, int dim_3_);
	void setWeight(const Tensor<double>& weight_);
	void setBias(const double& bias_);
	Tensor<double> getWeight();
	double getBias();
private:
	Tensor<double> weight;
	double bias;
	double clip;
};