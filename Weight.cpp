#include "Weight.h"

Weight::Weight() 
{ 
	bias = 0.f;
}

Weight::~Weight()
{

}

void Weight::initWeight(OPT_INIT weightInit_, int dim_0_, int dim_1_)
{
	clip = 1.f;
	if (weightInit_ == OPT_INIT::RANDOM)
	{
		clip = 1.f / (dim_0_ + dim_1_);
	}
	else if (weightInit_ == OPT_INIT::XAIVER)
	{
		clip = sqrt(6.f / (dim_0_ + dim_1_));
	}
	else if (weightInit_ == OPT_INIT::HE)
	{
		clip = sqrt(6.f / (dim_0_));
	}
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<double> dis(-clip, clip);
	weight.resize(dim_0_, dim_1_);
	int size = weight.size();
	for (int i = 0; i < size; i++)weight(i) = dis(gen);
	bias = dis(gen);
}

void Weight::initWeight(OPT_INIT weightInit_, int dim_0_, int dim_1_, int dim_2_, int dim_3_)
{
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<double> dis(-0.1, 0.1);
	weight.resize(dim_0_, dim_1_, dim_2_, dim_3_);
	int size = weight.size();
	for (int i = 0; i < size; i++)weight(i) = dis(gen);
	bias = dis(gen);
}

void Weight::setWeight(const Tensor<double>& weight_)
{ 
	weight = weight_;
	weight.clip(clip);
}

void Weight::setBias(const double& bias_) 
{ 
	bias = bias_; 
	if (bias > clip)
	{
		bias = clip;
	}
	if (bias < -clip)
	{
		bias = -clip;
	}
}

Tensor<double> Weight::getWeight() 
{ 
	return weight; 
}

double Weight::getBias()
{ 
	return bias; 
}