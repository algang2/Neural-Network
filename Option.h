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
	HE = 2
};

enum class OPT_ACTF
{
	LINEAR = 0,
	SIGMOID = 1,
	TANSIG = 2,
	RELU = 3,
	SOFTMAX =4
};

enum class OPT_LOSF
{
	SSE = 0,
	CROSSENTROPY = 1
};

enum class OPT_OPTM
{
	GDM = 0,
	RMSPROP = 1,
	ADAGRAD = 2,
	ADAM = 3
};

enum class OPT_LYR
{
	INPUT = 0,
	FULLYCONNECTED = 1,
	CONVOLUTION = 2,
	POOLING = 3
};
