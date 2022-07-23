#include "Weight.h"

Weight::Weight() 
{ 
	bias = 0.f;
}

Weight::~Weight()
{

}

void Weight::initWeight(OPT_INIT weightInit_, int dim_0_, int dim_1_, int dim_2_, int dim_3_)
{
	weight.resize(dim_0_, dim_1_, dim_2_, dim_3_);
	bias = 0.f;
	double clip = 1.f;
	if (weightInit_ == OPT_INIT::RANDOM)
	{
		clip = 1.f / (dim_0_ + dim_1_ + dim_2_ + dim_3_);
	}
	else if (weightInit_ == OPT_INIT::XAIVER)
	{
		clip = sqrt(6.f / (dim_0_ * dim_2_ * dim_3_ + dim_1_ * dim_2_ * dim_3_));
	}
	else if (weightInit_ == OPT_INIT::HE)
	{
		clip = sqrt(2.f / (dim_0_ * dim_2_ * dim_3_));
	}
	else if (weightInit_ == OPT_INIT::NONE)
	{
		return;
	}
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<double> dis(-clip, clip);
	int size = weight.size();
	for (int i = 0; i < size; i++)weight(i) = dis(gen);
	bias = dis(gen);
}

void Weight::setWeight(const Tensor<double>& weight_)
{ 
	weight = weight_;
}

void Weight::setBias(const double& bias_) 
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

std::string Weight::saveWeight()
{
	BinaryReader reader;
	int size = weight.size();
	reader.setNext(std::to_string(size), true);
	reader.setNext(std::to_string(weight.dim(0)));
	reader.setNext(std::to_string(weight.dim(1)));
	reader.setNext(std::to_string(weight.dim(2)));
	reader.setNext(std::to_string(weight.dim(3)));
	for (int i = 0; i < size; i++)
	{
		reader.setNext(std::to_string(weight.element(i)));
	}
	reader.setNext(std::to_string(bias));
	return reader.getBinary();
}

void Weight::loadWeight(BinaryReader& reader_)
{
	int size = stoi(reader_.getNext());
	int dim_0 = stoi(reader_.getNext());
	int dim_1 = stoi(reader_.getNext());
	int dim_2 = stoi(reader_.getNext());
	int dim_3 = stoi(reader_.getNext());
	weight.resize(dim_0, dim_1, dim_2, dim_3);
	for (int i = 0; i < size; i++)
	{
		weight(i) = stod(reader_.getNext());
	}
	bias = stod(reader_.getNext());
}