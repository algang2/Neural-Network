#pragma once
#include <math.h>

#include "Option.h"
#include "Tensor.h"

class Optimizer
{
public:
	virtual Tensor<double> optimizeDeltaWeight(const Tensor<double>& errors_, const Tensor<double>& prevNodes_) = 0;
	virtual double optimizeDeltaBias(const Tensor<double>& errors_) = 0;
protected:
	double learningRate;
};

class Null :public Optimizer
{
public:
	Null(const double& learningRate_);
	~Null();
	virtual Tensor<double> optimizeDeltaWeight(const Tensor<double>& errors_, const Tensor<double>& prevNodes_) override;
	virtual double optimizeDeltaBias(const Tensor<double>& errors_) override;
private:
	Tensor<double> opt_w;
	double opt_b;
};

class GDM :public Optimizer
{
public:
	GDM(const double& learningRate_, const Tensor<double>& weight_, const double& momentum_ = 0.99);
	~GDM();
	virtual Tensor<double> optimizeDeltaWeight(const Tensor<double>& errors_, const Tensor<double>& prevNodes_) override;
	virtual double optimizeDeltaBias(const Tensor<double>& errors_) override;
private:
	Tensor<double> opt_w;
	double opt_b;
	double momentum;
};

class RMSProp :public Optimizer
{
public:
	RMSProp(const double& learningRate_, const Tensor<double>& weight_, const double& gamma_ = 0.9);
	~RMSProp();
	virtual Tensor<double> optimizeDeltaWeight(const Tensor<double>& errors_, const Tensor<double>& prevNodes_) override;
	virtual double optimizeDeltaBias(const Tensor<double>& errors_) override;
private:
	Tensor<double> opt_w;
	double opt_b;
	double gamma;
};

class Adagrad :public Optimizer
{
public:
	Adagrad(const double& learningRate_, const Tensor<double>& weight_);
	~Adagrad();
	virtual Tensor<double> optimizeDeltaWeight(const Tensor<double>& errors_, const Tensor<double>& prevNodes_) override;
	virtual double optimizeDeltaBias(const Tensor<double>& errors_) override;
private:
	Tensor<double> opt_w;
	double opt_b;
};

class Adam :public Optimizer
{
public:
	Adam(const double& learningRate_, const Tensor<double>& weight_, const double& beta_1_ = 0.9, const double& beta_2_ = 0.999);
	~Adam();
	virtual Tensor<double> optimizeDeltaWeight(const Tensor<double>& errors_, const Tensor<double>& prevNodes_) override;
	virtual double optimizeDeltaBias(const Tensor<double>& errors_) override;
private:
	Tensor<double> opt_w_1;
	double opt_b_1;
	Tensor<double> opt_w_2;
	double opt_b_2;
	double beta_1;
	double beta_2;
	double beta_1_pow;
	double beta_2_pow;
};