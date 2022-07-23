#include "LayerInput.h"

LayerInput::LayerInput(int node_)
{
	dim[0] = node_;
	dim[1] = 1;
	dim[2] = 1;
	type = OPT_LYR::INPUT;
}

LayerInput::LayerInput(int depth_, int height_, int width_)
{
	dim[0] = depth_;
	dim[1] = height_;
	dim[2] = width_;
	type = OPT_LYR::INPUT;
}

LayerInput::LayerInput(BinaryReader& reader_)
{
	type = OPT_LYR::INPUT;
	dim[0] = stoi(reader_.getNext());
	dim[1] = stoi(reader_.getNext());
	dim[2] = stoi(reader_.getNext());
	actF = (OPT_ACTF)stoi(reader_.getNext());
	setActFunction(actF);
}

LayerInput::~LayerInput()
{

}

Tensor<double> LayerInput::forwardProp(const Tensor<double>& x_)
{
	nodes = x_;
	return x_;
}

Tensor<double> LayerInput::backwardProp(const Tensor<double>& e_in_)
{
	return e_in_;
}

void LayerInput::updateWeight()
{

}

std::string LayerInput::saveLayer()
{
	BinaryReader reader;
	reader.setNext(std::to_string((int)type), true);
	reader.setNext(std::to_string(dim[0]));
	reader.setNext(std::to_string(dim[1]));
	reader.setNext(std::to_string(dim[2]));
	reader.setNext(std::to_string((int)actF));
	return reader.getBinary();
}