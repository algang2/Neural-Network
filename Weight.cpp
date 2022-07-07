#include "Weight.h"

Weight::Weight() 
{ 
	bias = 0.f;
}

Weight::~Weight()
{

}

void Weight::initWeight(int dim_0_, int dim_1_)
{
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<double> dis(-0.1, 0.1);
	weight.resize(dim_0_, dim_1_);
	int size = weight.size();
	for (int i = 0; i < size; i++)weight(i) = dis(gen);
	bias = dis(gen);
}

void Weight::initWeight(int dim_0_, int dim_1_, int dim_2_, int dim_3_)
{
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<double> dis(-0.1, 0.1);
	weight.resize(dim_0_, dim_1_, dim_2_, dim_3_);
	int size = weight.size();
	for (int i = 0; i < size; i++)weight(i) = dis(gen);
	bias = dis(gen);
}

void Weight::setWeight(Tensor<double> weight_)
{ 
	weight = weight_; 
}

void Weight::setBias(double bias_) 
{ 
	bias = bias_; 
}

Tensor<double> Weight::getWeight() 
{ 
	return weight; 
}

double Weight::getBias()
{ 
	return bias; 
}