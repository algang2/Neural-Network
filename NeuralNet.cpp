#include "NeuralNet.h"
#include <stdio.h>

NeuralNet::NeuralNet(int epoch_, int batch_, OPT_OPTM optimizer_, OPT_LOSF lossF_, OPT_INIT weightInit_, OPT_NORM norm_, double rmse_)
	:epoch(epoch_), batch(batch_), optimizer(optimizer_), lossF(lossF_), weightInit(weightInit_), norm(norm_), rmse(rmse_)
{
	layerNum = 0;
}

NeuralNet::~NeuralNet()
{

}

void NeuralNet::setInput(const Tensor<double>& input_)
{
	input = input_;
}

void NeuralNet::setTarget(const Tensor<double>& target_)
{
	target = target_;
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

void NeuralNet::train()
{
	if (norm == OPT_NORM::USENORM)
	{
		//normalize
	}
	for (int ep = 0; ep < epoch; ep++)
	{
		double mse = 0.f;
		int batchSize = input.dim(0) / batch;
		for (int bat = 0; bat < batchSize; bat++)
		{
			Tensor<double> batchSliceX = sliceByBatch(input, bat * batch, (bat + 1) * batch);
			Tensor<double> batchSliceT = sliceByBatch(target, bat * batch, (bat + 1) * batch);
			Tensor<double> batchSliceY = forwardProp(batchSliceX);
			
			int dim_0 = batchSliceY.dim(0);
			int dim_1 = batchSliceY.dim(1);
			for (int i = 0; i < dim_0; i++)
			{
				for (int j = 0; j < dim_1; j++)
				{
					printf("%f[%f]\t", batchSliceY(i,j), batchSliceT(i, j));
				}
				printf("\n");
			}
			Tensor<double> batchSliceE = batchSliceT - batchSliceY;
			backwardProp(batchSliceE);
			updateWeight();
		}
		if (sqrt(mse) < rmse)break;
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