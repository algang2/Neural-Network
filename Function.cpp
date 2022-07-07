#include "Function.h"

Tensor<double> Linear::activation(const Tensor<double>& ten_)
{
	return ten_;
}

Tensor<double> Linear::deactivation(const Tensor<double>& ten_)
{
	Tensor<double> ten(ten_.dim(0), ten_.dim(1), ten_.dim(2), ten_.dim(3));
	ten = 1.f;
	return ten;
}

Tensor<double> Sigmoid::activation(const Tensor<double>& ten_)
{
	int size = ten_.size();
	Tensor<double> ten(ten_);
	for (int i = 0; i < size; i++)ten(i) = 1.f / (1.f + exp(-ten_.element(i)));
	return ten;
}

Tensor<double> Sigmoid::deactivation(const Tensor<double>& ten_)
{
	int size = ten_.size();
	Tensor<double> ten(ten_);
	for (int i = 0; i < size; i++)ten(i) = ten_.element(i) * (1 - ten_.element(i));
	return ten;
}

Tensor<double> Tansig::activation(const Tensor<double>& ten_)
{
	int size = ten_.size();
	Tensor<double> ten(ten_);
	for (int i = 0; i < size; i++)ten(i) = 2 / (1 + exp(-2 * ten_.element(i))) - 1;
	return ten;
}

Tensor<double> Tansig::deactivation(const Tensor<double>& ten_)
{
	int size = ten_.size();
	Tensor<double> ten(ten_);
	for (int i = 0; i < size; i++)ten(i) = 1 - pow(ten_.element(i), 2);
	return ten;
}

Tensor<double> ReLU::activation(const Tensor<double>& ten_)
{
	int size = ten_.size();
	Tensor<double> ten(ten_);
	for (int i = 0; i < size; i++)
	{
		if (ten_.element(i) > 0) ten(i) = ten_.element(i);
		else ten(i) = 0;
	}
	return ten;
}

Tensor<double> ReLU::deactivation(const Tensor<double>& ten_)
{
	int size = ten_.size();
	Tensor<double> ten(ten_);
	for (int i = 0; i < size; i++)
	{
		if (ten_.element(i) > 0) ten(i) = 1;
		else ten(i) = 0;
	}
	return ten;
}

Tensor<double> Softmax::activation(const Tensor<double>& ten_)
{
	int dim_0 = ten_.dim(0);
	int dim_1 = ten_.dim(1);
	Tensor<double> ten(ten_);
	for (int i = 0; i < dim_0; i++)
	{	
		double sum = 0.f;
		for (int j = 0; j < dim_1; j++)
		{
			sum += exp(ten_.element(i, j));
		}
		for (int j = 0; j < dim_1; j++)
		{
			ten(i,j) = exp(ten_.element(i, j)) / sum;
		}
	}
	return ten;
}

Tensor<double> Softmax::deactivation(const Tensor<double>& ten_)
{
	Tensor<double> ten(ten_.dim(0), ten_.dim(1), ten_.dim(2), ten_.dim(3));
	ten = 1.f;
	return ten;
}