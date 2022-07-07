#pragma once
#include <random>
#include "Tensor.h"

class Weight
{
public:
	Weight();
	~Weight();
	void initWeight(int dim_0_, int dim_1_);
	void initWeight(int dim_0_, int dim_1_, int dim_2_, int dim_3_);
	void setWeight(Tensor<double> weight_);
	void setBias(double bias_);
	Tensor<double> getWeight();
	double getBias();
private:
	Tensor<double> weight;
	double bias;
};