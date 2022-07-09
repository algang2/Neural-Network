#pragma once
#include <map>
#include <vector>
#include <string>
#include <stdio.h>
#include "Layer.h"

class NeuralNet
{
public:
	NeuralNet(int epoch_, int batch_, OPT_LRN learn_, OPT_OPTM optimizer_, OPT_LOSF lossF_, OPT_INIT weightInit_, OPT_NORM norm_ = OPT_NORM::USENORM, double rmse_ = 0.f);
	~NeuralNet();
	void setInput(const Tensor<std::string>& input_);
	void setTarget(const Tensor<std::string>& target_);
	void addLayer(OPT_LYR layer_, int node_, OPT_ACTF actF_ = OPT_ACTF::LINEAR);
	Tensor<std::string> train();
	Tensor<std::string> predict();
	Tensor<double> forwardProp(const Tensor<double>& x_);
	void backwardProp(const Tensor<double>& e_);
	void updateWeight();
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
	OPT_OPTM optimizer;
	OPT_LOSF lossF;
	OPT_INIT weightInit;
	OPT_NORM norm;
	double rmse;

	Tensor<double> input;
	Tensor<double> target;

	int layerNum;
	std::vector<Layer*> layer;
	std::map<std::string, int> encoder;
};