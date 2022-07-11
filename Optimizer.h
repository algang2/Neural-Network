#pragma once
#include "Option.h"
#include "Tensor.h"

class Optimizer
{
public:
	virtual Tensor<double> optimizeDeltaWeight(Tensor<double>d_weight_) = 0;
	virtual double optimizeDeltaBias(const double& d_bias_) = 0;
protected:
	double learningRate;
};

class Null :public Optimizer
{
public:
	Null(const double& learningRate_);
	~Null();
	virtual Tensor<double> optimizeDeltaWeight(Tensor<double>d_weight_) override;
	virtual double optimizeDeltaBias(const double& d_bias_) override;
private:
	Tensor<double> opt_w;
	double opt_b;
};

class GDM :public Optimizer
{
public:
	GDM(const double& learningRate_, const double& momentum_, const Tensor<double>& weight_);
	~GDM();
	virtual Tensor<double> optimizeDeltaWeight(Tensor<double>d_weight_) override;
	virtual double optimizeDeltaBias(const double& d_bias_) override;
private:
	Tensor<double> opt_w;
	double opt_b;
	double momentum;
};

class RMSProp :public Optimizer
{
public:
	virtual Tensor<double> optimizeDeltaWeight(Tensor<double>d_weight_) override;
	virtual double optimizeDeltaBias(const double& d_bias_) override;
private:
	Tensor<double> opt_w;
	double opt_b;
	double gamma;
};

class Adagrad :public Optimizer
{
public:
	virtual Tensor<double> optimizeDeltaWeight(Tensor<double>d_weight_) override;
	virtual double optimizeDeltaBias(const double& d_bias_) override;
private:
	Tensor<double> opt_w;
	double opt_b;
};

class Adam :public Optimizer
{
public:
	virtual Tensor<double> optimizeDeltaWeight(Tensor<double>d_weight_) override;
	virtual double optimizeDeltaBias(const double& d_bias_) override;
private:
	Tensor<double> opt_w_1;
	double opt_b_1;
	Tensor<double> opt_w_2;
	double opt_b_2;
	double beta_1;
	double beta_2;
};