#include "Optimizer.h"

Null::Null(const double& learningRate_)
{
	learningRate = learningRate_;
}

Null::~Null()
{

}

Tensor<double> Null::optimizeDeltaWeight(const Tensor<double>& errors_, const Tensor<double>& prevNodes_)
{
	int dataNum = errors_.dim(0);
	int prevNodeNum = opt_w.dim(0);
	int postNodeNum = opt_w.dim(1);
	for (int i = 0; i < prevNodeNum; i++)
	{
		for (int j = 0; j < postNodeNum; j++)
		{
			double sum = 0.f;
			for (int k = 0; k < dataNum; k++)
			{
				sum += errors_.element(k, j) * prevNodes_.element(k, i);
			}
			sum = sum / dataNum;
			opt_w(i, j) = - learningRate * sum;
		}
	}
	return opt_w;
}

double Null::optimizeDeltaBias(const Tensor<double>& errors_)
{
	int dataNum = errors_.dim(0);
	int postNodeNum = opt_w.dim(1);
	double sum = 0.f;
	for (int j = 0; j < postNodeNum; j++)
	{
		for (int k = 0; k < dataNum; k++)
		{
			sum += errors_.element(k, j);
		}	
	}
	sum = sum / dataNum;
	opt_b = -learningRate * sum;
	return opt_b;
}

GDM::GDM(const double& learningRate_, const Tensor<double>& weight_, const double& momentum_)
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

Tensor<double> GDM::optimizeDeltaWeight(const Tensor<double>& errors_, const Tensor<double>& prevNodes_)
{
	int dataNum = errors_.dim(0);
	int prevNodeNum = opt_w.dim(0);
	int postNodeNum = opt_w.dim(1);
	for (int i = 0; i < prevNodeNum; i++)
	{
		for (int j = 0; j < postNodeNum; j++)
		{
			double sum = 0.f;
			for (int k = 0; k < dataNum; k++)
			{
				sum += errors_.element(k, j) * prevNodes_.element(k, i);
			}
			sum = sum / dataNum;
			opt_w(i, j) = momentum * opt_w(i, j) - learningRate * sum;
		}
	}
	return opt_w;
}

double GDM::optimizeDeltaBias(const Tensor<double>& errors_)
{
	int dataNum = errors_.dim(0);
	int postNodeNum = opt_w.dim(1);
	double sum = 0.f;
	for (int j = 0; j < postNodeNum; j++)
	{
		for (int k = 0; k < dataNum; k++)
		{
			sum += errors_.element(k, j);
		}
	}
	sum = sum / dataNum;
	opt_b = momentum * opt_b - learningRate * sum;
	return opt_b;
}

RMSProp::RMSProp(const double& learningRate_, const Tensor<double>& weight_, const double& gamma_)
{
	learningRate = learningRate_;
	gamma = gamma_;
	int dim_0 = weight_.dim(0);
	int dim_1 = weight_.dim(1);
	int dim_2 = weight_.dim(2);
	int dim_3 = weight_.dim(3);
	opt_w.resize(dim_0, dim_1, dim_2, dim_3);
	opt_b = 0.f;
}

RMSProp::~RMSProp()
{

}

Tensor<double> RMSProp::optimizeDeltaWeight(const Tensor<double>& errors_, const Tensor<double>& prevNodes_)
{
	int dataNum = errors_.dim(0);
	int prevNodeNum = opt_w.dim(0);
	int postNodeNum = opt_w.dim(1);
	Tensor<double> delta_w(opt_w);
	double epsilon = 1e-10;
	for (int i = 0; i < prevNodeNum; i++)
	{
		for (int j = 0; j < postNodeNum; j++)
		{
			double sum = 0.f;
			for (int k = 0; k < dataNum; k++)
			{
				sum += errors_.element(k, j) * prevNodes_.element(k, i);
			}
			sum = sum / dataNum;
			opt_w(i, j) = gamma * opt_w(i, j) + (1 - gamma) * sum * sum;
			delta_w(i, j) = -learningRate / sqrt(opt_w(i, j) + epsilon) * sum;
		}
	}
	return delta_w;
}

double RMSProp::optimizeDeltaBias(const Tensor<double>& errors_)
{
	int dataNum = errors_.dim(0);
	int postNodeNum = opt_w.dim(1);
	double delta_b = 0.f;
	double epsilon = 1e-10;
	double sum = 0.f;
	for (int j = 0; j < postNodeNum; j++)
	{
		for (int k = 0; k < dataNum; k++)
		{
			sum += errors_.element(k, j);
		}
	}
	sum = sum / dataNum;
	opt_b = gamma * opt_b + (1 - gamma) * sum * sum;
	delta_b = -learningRate / sqrt(opt_b + epsilon) * sum;
	return delta_b;
}

Adagrad::Adagrad(const double& learningRate_, const Tensor<double>& weight_)
{
	learningRate = learningRate_;
	int dim_0 = weight_.dim(0);
	int dim_1 = weight_.dim(1);
	int dim_2 = weight_.dim(2);
	int dim_3 = weight_.dim(3);
	opt_w.resize(dim_0, dim_1, dim_2, dim_3);
	opt_b = 0.f;
}

Adagrad::~Adagrad()
{

}

Tensor<double> Adagrad::optimizeDeltaWeight(const Tensor<double>& errors_, const Tensor<double>& prevNodes_)
{
	int dataNum = errors_.dim(0);
	int prevNodeNum = opt_w.dim(0);
	int postNodeNum = opt_w.dim(1);
	Tensor<double> delta_w(opt_w);
	double epsilon = 1e-10;
	for (int i = 0; i < prevNodeNum; i++)
	{
		for (int j = 0; j < postNodeNum; j++)
		{
			double sum = 0.f;
			for (int k = 0; k < dataNum; k++)
			{
				sum += errors_.element(k, j) * prevNodes_.element(k, i);
			}
			sum = sum / dataNum;
			opt_w(i, j) = opt_w(i, j) + sum * sum;
			delta_w(i, j) = -learningRate / sqrt(opt_w(i, j) + epsilon) * sum;
		}
	}
	return delta_w;
}

double Adagrad::optimizeDeltaBias(const Tensor<double>& errors_)
{
	int dataNum = errors_.dim(0);
	int postNodeNum = opt_w.dim(1);
	double delta_b = 0.f;
	double epsilon = 1e-10;
	double sum = 0.f;
	for (int j = 0; j < postNodeNum; j++)
	{
		for (int k = 0; k < dataNum; k++)
		{
			sum += errors_.element(k, j);
		}
	}
	sum = sum / dataNum;
	opt_b = opt_b + sum * sum;
	delta_b = -learningRate / sqrt(opt_b + epsilon) * sum;
	return delta_b;
}

Adam::Adam(const double& learningRate_, const Tensor<double>& weight_, const double& beta_1_, const double& beta_2_)
{
	learningRate = learningRate_;
	beta_1 = beta_1_;
	beta_2 = beta_2_;
	int dim_0 = weight_.dim(0);
	int dim_1 = weight_.dim(1);
	int dim_2 = weight_.dim(2);
	int dim_3 = weight_.dim(3);
	opt_w_1.resize(dim_0, dim_1, dim_2, dim_3);
	opt_b_1 = 0.f;
	opt_w_2.resize(dim_0, dim_1, dim_2, dim_3);
	opt_b_2 = 0.f;
	beta_1_pow = 1.f;
	beta_2_pow = 1.f;
}

Adam::~Adam()
{

}

Tensor<double> Adam::optimizeDeltaWeight(const Tensor<double>& errors_, const Tensor<double>& prevNodes_)
{
	int dataNum = errors_.dim(0);
	int prevNodeNum = opt_w_1.dim(0);
	int postNodeNum = opt_w_1.dim(1);
	Tensor<double> delta_w(opt_w_1);
	double epsilon = 1e-10;
	beta_1_pow *= beta_1;
	beta_2_pow *= beta_2;
	for (int i = 0; i < prevNodeNum; i++)
	{
		for (int j = 0; j < postNodeNum; j++)
		{
			double sum = 0.f;
			for (int k = 0; k < dataNum; k++)
			{
				sum += errors_.element(k, j) * prevNodes_.element(k, i);
			}
			sum = sum / dataNum;
			opt_w_1(i, j) = beta_1 * opt_w_1(i, j) + (1 - beta_1) * sum;
			opt_w_2(i, j) = beta_2 * opt_w_2(i, j) + (1 - beta_2) * sum * sum;
			double cor_w_1 = opt_w_1(i, j) / (1 - beta_1_pow);
			double cor_w_2 = opt_w_2(i, j) / (1 - beta_2_pow);
			delta_w(i, j) = -learningRate * cor_w_1 / sqrt(cor_w_2 + epsilon);
		}
	}
	return delta_w;
}

double Adam::optimizeDeltaBias(const Tensor<double>& errors_)
{
	int dataNum = errors_.dim(0);
	int postNodeNum = opt_w_1.dim(1);
	double delta_b = 0.f;
	double epsilon = 1e-10;
	double sum = 0.f;
	for (int j = 0; j < postNodeNum; j++)
	{
		for (int k = 0; k < dataNum; k++)
		{
			sum += errors_.element(k, j);
		}
	}
	sum = sum / dataNum;
	opt_b_1 = beta_1 * opt_b_1 + (1 - beta_1) * sum;
	opt_b_2 = beta_2 * opt_b_2 + (1 - beta_2) * sum * sum;
	double cor_b_1 = opt_b_1 / (1 - beta_1_pow);
	double cor_b_2 = opt_b_2 / (1 - beta_2_pow);
	delta_b = -learningRate * cor_b_1 / sqrt(cor_b_2 + epsilon);
	return delta_b;
}