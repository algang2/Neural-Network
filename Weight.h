#pragma once
#include <random>
#include "Option.h"
#include "Tensor.h"
#include "BinaryReader.h"

class Weight
{
public:
	Weight();
	~Weight();
	void initWeight(OPT_INIT weightInit_, int dim_0_, int dim_1_, int dim_2_ = 1, int dim_3_ = 1);
	void setWeight(const Tensor<double>& weight_);
	void setBias(const double& bias_);
	Tensor<double> getWeight();
	double getBias();
	std::string saveWeight();
	void loadWeight(BinaryReader& reader_);
private:
	Tensor<double> weight;
	double bias;
};