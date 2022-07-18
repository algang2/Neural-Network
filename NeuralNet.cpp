#include "NeuralNet.h"

NeuralNet::NeuralNet()
{
	error = false;
	trained = false;
	loaded = false;
	layerNum = 0;
}

NeuralNet::NeuralNet(OPT_LRN learn_, OPT_INIT weightInit_, OPT_NORM norm_, int epoch_, int batch_, double learningRate_)
	:learn(learn_), weightInit(weightInit_), norm(norm_), epoch(epoch_), batch(batch_), learningRate(learningRate_)
{
	error = false;
	trained = false;
	loaded = false;
	layerNum = 0;
}

NeuralNet::~NeuralNet()
{

}

void NeuralNet::setInput(const Tensor<std::string>& input_)
{
	if (input_.size() == 0)
	{
		error = true;
		printf("[Error]No input data.\n");
		return;
	}
	try
	{
		int dim_0 = input_.dim(0);
		int dim_1 = input_.dim(1);
		int dim_2 = input_.dim(2);
		int dim_3 = input_.dim(3);
		int size = input_.size();
		input.resize(dim_0, dim_1, dim_2, dim_3);
		for (int i = 0; i < size; i++)input(i) = std::stod(input_.element(i));
	}
	catch (...)
	{
		error = true;
		printf("[Error]Input data type must be numbers.\n");
		return;
	}
	if (norm == OPT_NORM::USENORM)
	{
		normalizeInput();
	}
}

void NeuralNet::setTarget(const Tensor<std::string>& target_)
{
	if (target_.size() == 0)
	{
		error = true;
		printf("[Error]No target data.\n");
		return;
	}
	try
	{
		if (learn == OPT_LRN::REGRESSION)
		{
			int size = target_.size();
			target.resize(size);
			for (int i = 0; i < size; i++)
			{
				target(i) = std::stod(target_.element(i));
			}
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
		error = true;
		printf("[Error]Target data type must be numbers.\n");
		return;
	}
}

void NeuralNet::addLayer(OPT_LYR layer_, int node_, OPT_ACTF actFnc_)
{
	Layer* prevlyrptr = nullptr;
	if(layerNum > 0) prevlyrptr = layer[layerNum - 1];
	Layer* lyrptr = Layer::initLayer(layer_, node_, weightInit, prevlyrptr);
	if (lyrptr == nullptr)
	{
		error = true;
		printf("[Error]Layer not available.\n");
		return;
	}
	lyrptr->setActFunction(actFnc_);
	if (optimizer == OPT_OPTM::GDM)
	{
		lyrptr->setOptimizer(optimizer, learningRate, momentum);
	}
	else if (optimizer == OPT_OPTM::RMSPROP)
	{
		lyrptr->setOptimizer(optimizer, learningRate, gamma);
	}
	else if (optimizer == OPT_OPTM::ADAGRAD)
	{
		lyrptr->setOptimizer(optimizer, learningRate);
	}
	else if (optimizer == OPT_OPTM::ADAM)
	{
		lyrptr->setOptimizer(optimizer, learningRate, beta_1, beta_2);
	}
	else
	{
		lyrptr->setOptimizer(optimizer, learningRate);
	}
	layer.push_back(lyrptr);
	layerNum++;
}

Tensor<std::string> NeuralNet::train()
{
	if (error)
	{
		printf("[Error]Failed training.\n");
		Tensor<std::string> retE(1);
		retE(0) = "[Error]Failed training.";
		error = false;
		return retE;
	}
	if (loaded)
	{
		printf("[Error]Loaded network cannot be trained.\n");
		Tensor<std::string> retE(1);
		retE(0) = "[Error]Loaded network cannot be trained.";
		error = false;
		return retE;
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
			Tensor<double> batchSliceE = batchSliceY - batchSliceT;
			backwardProp(batchSliceE);
			updateWeight();
			mse += calMSE(batchSliceE);
		}
		printf("##Epoch %d RMSE: %f##\n", ep + 1, sqrt(mse));
	}
	trained = true;
	return predict();
}

Tensor<std::string> NeuralNet::predict()
{
	if (!trained)
	{
		printf("[Error]Network not trained.\n");
		Tensor<std::string> retE(1);
		retE(0) = "[Error]Network not trained.";
		return retE;
	}
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

void NeuralNet::setOptimizer(OPT_OPTM optimizer_, const double& val_0_, const double& val_1_)
{
	optimizer = optimizer_;
	if (optimizer == OPT_OPTM::GDM)
	{
		momentum = val_0_;
		if (momentum == 0.f)
		{
			momentum = 0.99;
		}
		for (int i = 0; i < layerNum; i++)
		{
			layer[i]->setOptimizer(optimizer, learningRate, momentum);
		}
	}
	else if (optimizer == OPT_OPTM::RMSPROP)
	{
		gamma = val_0_;
		if (gamma == 0.f)
		{
			gamma = 0.9;
		}
		for (int i = 0; i < layerNum; i++)
		{
			layer[i]->setOptimizer(optimizer, learningRate, gamma);
		}
	}
	else if (optimizer == OPT_OPTM::ADAGRAD)
	{
		for (int i = 0; i < layerNum; i++)
		{
			layer[i]->setOptimizer(optimizer, learningRate);
		}
	}
	else if (optimizer == OPT_OPTM::ADAM)
	{
		beta_1 = val_0_;
		beta_2 = val_1_;
		if (beta_1 == 0.f)
		{
			beta_1 = 0.9;
		}
		if (beta_2 == 0.f)
		{
			beta_2 = 0.999;
		}
		for (int i = 0; i < layerNum; i++)
		{
			layer[i]->setOptimizer(optimizer, learningRate, beta_1, beta_2);
		}
	}
	else
	{
		for (int i = 0; i < layerNum; i++)
		{
			layer[i]->setOptimizer(optimizer, learningRate);
		}
	}
}

void NeuralNet::saveNetwork(std::string dir_)
{
	if (!trained)
	{
		printf("[Error]Network not trained.\n");
		return;
	}
	BinaryReader reader;
	reader.setNext(std::to_string((int)learn), true);
	reader.setNext(std::to_string((int)norm));
	reader.setNext(std::to_string(layerNum));
	for (int i = 0; i < layerNum; i++)
	{
		reader.setNext(std::to_string((int)layer[i]->type));
		reader.setNext(std::to_string(layer[i]->nodeNum));
		reader.setNext(std::to_string((int)(layer[i]->actF)));
		reader.setNext(layer[i]->weight.saveWeight());
	}
	reader.save(dir_ + "\\NeuralNet.mdl");
}

void NeuralNet::loadNetwork(std::string dir_)
{
	trained = true;
	loaded = true;
	weightInit = OPT_INIT::NONE;
	BinaryReader reader(dir_);
	learn = (OPT_LRN)stoi(reader.getNext());
	norm = (OPT_NORM)stoi(reader.getNext());
	int num = stoi(reader.getNext());
	for (int i = 0; i < num; i++)
	{
		OPT_LYR type = (OPT_LYR)stoi(reader.getNext());
		int nodeNum = stoi(reader.getNext());
		OPT_ACTF actF = (OPT_ACTF)stoi(reader.getNext());
		addLayer(type, nodeNum, actF);
		int size = stoi(reader.getNext());
		std::string weight;
		weight += std::to_string(size) + ",";
		for (int j = 0; j < size; j++)
		{
			weight += reader.getNext() + ",";
		}
		weight += reader.getNext();
		layer[i]->weight.loadWeight(weight);
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

void NeuralNet::normalizeInput()
{
	int dim_0 = input.dim(0);
	int dim_1 = input.dim(1);
	int dim_2 = input.dim(2);
	int dim_3 = input.dim(3);
	for (int j = 0; j < dim_1; j++)
	{
		for (int k = 0; k < dim_2; k++)
		{
			for (int l = 0; l < dim_3; l++)
			{
				double average = 0.f;
				double variance = 0.f;
				for (int i = 0; i < dim_0; i++)
				{
					average += input(i, j, k, l);
				}
				average /= dim_0;
				for (int i = 0; i < dim_0; i++)
				{
					variance += pow(input(i, j, k, l), 2);
				}
				variance /= dim_0;
				double standard = sqrt(variance);
				for (int i = 0; i < dim_0; i++)
				{
					input(i, j, k, l) = (input(i, j, k, l) - average) / standard;
				}
			}
		}
	}
}