#pragma once
#include <math.h>

#include "Tensor.h"
#include "Option.h"

class ActFunction
{
public:
	virtual Tensor<double> activation(const Tensor<double>& ten_) = 0;
	virtual Tensor<double> deactivation(const Tensor<double>& ten_) = 0;
};

class Linear: public ActFunction
{
	Tensor<double> activation(const Tensor<double>& ten_) override;
	Tensor<double> deactivation(const Tensor<double>& ten_) override;
};

class Sigmoid : public ActFunction
{
	Tensor<double> activation(const Tensor<double>& ten_) override;
	Tensor<double> deactivation(const Tensor<double>& ten_) override;
};

class Tansig: public ActFunction
{
	Tensor<double> activation(const Tensor<double>& ten_) override;
	Tensor<double> deactivation(const Tensor<double>& ten_) override;
};

class ReLU : public ActFunction
{
	Tensor<double> activation(const Tensor<double>& ten_) override;
	Tensor<double> deactivation(const Tensor<double>& ten_) override;
};

class Softmax : public ActFunction
{
	Tensor<double> activation(const Tensor<double>& ten_) override;
	Tensor<double> deactivation(const Tensor<double>& ten_) override;
};

class lossFuntion
{

};

class SSE : public lossFuntion
{

};

class CrossEntropy : public lossFuntion
{

};