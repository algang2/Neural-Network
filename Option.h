#pragma once

enum class OPT_LRN
{
	REGRESSION = 0,
	CLASSIFICATION = 1
};

enum class OPT_NORM
{
	USENORM = 0,
	NOUSENORM = 1
};

enum class OPT_INIT
{
	RANDOM = 0,
	XAIVER = 1,
	HE = 2,
	NONE = 3
};

enum class OPT_ACTF
{
	LINEAR = 0,
	SIGMOID = 1,
	TANSIG = 2,
	RELU = 3,
	LEAKYRELU = 4,
	SOFTMAX =5
};

enum class OPT_OPTM
{
	GDM = 1,
	RMSPROP = 2,
	ADAGRAD = 3,
	ADAM = 4
};

enum class OPT_LYR
{
	INPUT = 0,
	FULLYCONNECTED = 1,
	CONVOLUTION = 2,
	POOLING = 3
};
