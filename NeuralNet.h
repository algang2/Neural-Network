#pragma once
#include <map>
#include <vector>
#include <string>
#include <stdio.h>
#include "Layer.h"

class NeuralNet
{
public:
	NeuralNet(OPT_LRN learn_, OPT_INIT weightInit_, OPT_NORM norm_, int epoch_, int batch_, double learningRate_ = 0.001, double rmse_ = 0.f);
	~NeuralNet();
	void setInput(const Tensor<std::string>& input_);
	void setTarget(const Tensor<std::string>& target_);
	void addLayer(OPT_LYR layer_, int node_, OPT_ACTF actFnc_ = OPT_ACTF::LINEAR);
	Tensor<std::string> train();
	Tensor<std::string> predict();
	Tensor<double> forwardProp(const Tensor<double>& x_);
	void backwardProp(const Tensor<double>& e_);
	void updateWeight();
	void setOptimizer(OPT_OPTM optimizer_, const double& val_0_ = 0.f, const double& val_1_ = 0.f);
private: 
	double calMSE(const Tensor<double>& error_);
	Tensor<double> sliceByBatch(const Tensor<double>& ten_, const int& prevIdx_, const int& postIdx_);
	void oneHotEncoding(const Tensor<std::string>& target_);
	Tensor<std::string> oneHotDecoding(const Tensor<double>& output_);
	void normalization();
private:
	int epoch;
	int batch;
	OPT_LRN learn;
	OPT_INIT weightInit;
	OPT_NORM norm;
	double rmse;

	Tensor<double> input;
	Tensor<double> target;

	bool error;
	int layerNum;
	std::vector<Layer*> layer;
	std::map<std::string, int> encoder;

	OPT_OPTM optimizer;
	double learningRate;
	double momentum;
	double gamma;
	double beta_1;
	double beta_2;
};