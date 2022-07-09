#include "NeuralNet.h"

NeuralNet::NeuralNet(int epoch_, int batch_, OPT_LRN learn_, OPT_OPTM optimizer_, OPT_LOSF lossF_, OPT_INIT weightInit_, OPT_NORM norm_, double rmse_)
	:epoch(epoch_), batch(batch_), learn(learn_), optimizer(optimizer_), lossF(lossF_), weightInit(weightInit_), norm(norm_), rmse(rmse_)
{
	layerNum = 0;
}

NeuralNet::~NeuralNet()
{

}

void NeuralNet::setInput(const Tensor<std::string>& input_)
{
	try
	{
		int dim_0 = input_.dim(0);
		int dim_1 = input_.dim(1);
		int dim_2 = input_.dim(2);
		int dim_3 = input_.dim(3);
		int size = input_.size();
		input.resize(dim_0, dim_1, dim_2, dim_3);
		for (int i = 0; i < size; i++)input(i) = std::stod(input_.element(i));
		if (norm == OPT_NORM::USENORM)
		{
			normalization();
		}
	}
	catch (...)
	{
		printf("#Input data type must be Numbers.");
	}
}

void NeuralNet::setTarget(const Tensor<std::string>& target_)
{
	try
	{
		if (learn == OPT_LRN::REGRESSION)
		{
			int size = target_.size();
			target.resize(size);
			for (int i = 0; i < size; i++)target(i) = std::stod(target_.element(i));
		}
		else if (learn == OPT_LRN::CLASSIFICATION)
		{
			oneHotEncoding(target_);
			int size = target_.size();
			int encode = encoder.size();
			target.resize(size, encode);
			for (int i = 0; i < size; i++)
			{
				target(i, encoder[target_.element(i)]) = 1;
			}
		}
	}
	catch (...)
	{
		printf("#Target data type must be Numbers.");
	}
}

void NeuralNet::addLayer(OPT_LYR layer_, int node_, OPT_ACTF actF_)
{
	Layer* prevlyrptr = nullptr;
	if(layerNum > 0) prevlyrptr = layer[layerNum - 1];
	Layer* lyrptr = Layer::initLayer(layer_, node_, prevlyrptr);
	lyrptr->setActF(actF_);
	layer.push_back(lyrptr);
	layerNum++;
}

Tensor<std::string> NeuralNet::train()
{
	for (int ep = 0; ep < epoch; ep++)
	{
		double mse = 0.f;
		int batchSize = input.dim(0) / batch;
		for (int bat = 0; bat < batchSize; bat++)
		{
			Tensor<double> batchSliceX = sliceByBatch(input, bat * batch, (bat + 1) * batch);
			Tensor<double> batchSliceT = sliceByBatch(target, bat * batch, (bat + 1) * batch);
			Tensor<double> batchSliceY = forwardProp(batchSliceX);
			Tensor<double> batchSliceE = batchSliceT - batchSliceY;
			backwardProp(batchSliceE);
			updateWeight();
			mse += calMSE(batchSliceE);
		}
		printf("##Epoch %d RMSE: %f##\n", ep, sqrt(mse));
		if (sqrt(mse) < rmse)break;
	}
	return predict();
}

Tensor<std::string> NeuralNet::predict()
{
	Tensor<double> output = forwardProp(input);
	if (learn == OPT_LRN::REGRESSION)
	{
		int size = output.size();
		Tensor<std::string> outputStr(size);
		for (int i = 0; i < size; i++)outputStr(i) = std::to_string(output.element(i));
		return outputStr;
	}
	else if (learn == OPT_LRN::CLASSIFICATION)
	{
		return oneHotDecoding(output);
	}
}

Tensor<double> NeuralNet::forwardProp(const Tensor<double>& x_)
{
	Tensor<double> y = x_;
	for (int i = 0; i < layerNum; i++)
	{
		y = layer[i]->forwardProp(y);
	}
	return y;
}

void NeuralNet::backwardProp(const Tensor<double>& e_)
{
	Tensor<double> e = e_;
	for (int i = layerNum - 1; i >= 0; i--)
	{
		e = layer[i]->backwardProp(e);
	}
}

void NeuralNet::updateWeight()
{
	for (int i = 0; i < layerNum; i++)
	{
		layer[i]->updateWeight();
	}
}

double NeuralNet::calMSE(const Tensor<double>& error_)
{
	int size = error_.size();
	double mse = 0.f;
	for (int i = 0; i < size; i++)mse += error_.element(i) * error_.element(i);
	return mse;
}

Tensor<double> NeuralNet::sliceByBatch(const Tensor<double>& ten_, const int& prevIdx_, const int& postIdx_)
{
	Tensor<double> tmpTen(postIdx_ - prevIdx_, ten_.dim(1), ten_.dim(2), ten_.dim(3));
	int dim_0 = tmpTen.dim(0);
	int dim_1 = tmpTen.dim(1);
	int dim_2 = tmpTen.dim(2);
	int dim_3 = tmpTen.dim(3);

	for (int i = 0; i < dim_0; i++)
	{
		for (int j = 0; j < dim_1; j++)
		{
			for (int k = 0; k < dim_2; k++)
			{
				for (int l = 0; l < dim_3; l++)
				{
					tmpTen(i, j, k, l) = ten_.element(prevIdx_ + i, j, k, l);
				}
			}
		}
	}
	return tmpTen;
}

void NeuralNet::oneHotEncoding(const Tensor<std::string>& target_)
{
	int size = target_.size();
	for (int i = 0; i < size; i++)
	{
		encoder[target_.element(i)];
	}
	for (int i = 0; i < encoder.size(); i++)
	{
		encoder[target_.element(i)] = i;
	}
}

Tensor<std::string> NeuralNet::oneHotDecoding(const Tensor<double>& output_)
{
	int size = output_.dim(0);
	int encode = output_.dim(1);
	Tensor<std::string> outputStr(size);
	for (int i = 0; i < size; i++)
	{
		double max = 0.f;
		for (auto iter = encoder.begin(); iter != encoder.end(); iter++)
		{
			int pos = iter->second;
			if (output_.element(i, pos) > max)
			{
				max = output_.element(i, pos);
				outputStr(i) = iter->first;
			}
		}
	}
	return outputStr;
}

void NeuralNet::normalization()
{
	
}