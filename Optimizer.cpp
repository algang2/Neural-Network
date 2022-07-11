#include "Optimizer.h"

Null::Null(const double& learningRate_)
{
	learningRate = learningRate_;
}

Null::~Null()
{

}

Tensor<double> Null::optimizeDeltaWeight(Tensor<double>d_weight_)
{
	opt_w = -learningRate * d_weight_;
	return opt_w;
}

double Null::optimizeDeltaBias(const double& d_bias_)
{
	opt_b = -learningRate * d_bias_;
	return opt_b;
}

GDM::GDM(const double& learningRate_, const double& momentum_, const Tensor<double>& weight_)
{
	learningRate = learningRate_;
	momentum = momentum_;
	int dim_0 = weight_.dim(0);
	int dim_1 = weight_.dim(1);
	int dim_2 = weight_.dim(2);
	int dim_3 = weight_.dim(3);
	opt_w.resize(dim_0, dim_1, dim_2, dim_3);
	opt_b = 0.f;
}

GDM::~GDM()
{

}

Tensor<double> GDM::optimizeDeltaWeight(Tensor<double>d_weight_)
{
	opt_w = momentum * opt_w - learningRate * d_weight_;
	return opt_w;
}

double GDM::optimizeDeltaBias(const double& d_bias_)
{
	opt_b = momentum * opt_b - learningRate * d_bias_;
	return opt_b;
}














Tensor<double> RMSProp::optimizeDeltaWeight(Tensor<double>d_weight_)
{
	return opt_w;
}

double RMSProp::optimizeDeltaBias(const double& d_bias_)
{
	return opt_b;
}

Tensor<double> Adagrad::optimizeDeltaWeight(Tensor<double>d_weight_)
{
	return opt_w;
}

double Adagrad::optimizeDeltaBias(const double& d_bias_)
{
	return opt_b;
}

Tensor<double> Adam::optimizeDeltaWeight(Tensor<double>d_weight_)
{
	return opt_w_1;
}

double Adam::optimizeDeltaBias(const double& d_bias_)
{
	return opt_b_1;
}