#include <iostream>
#include "NeuralNet.h"
#include "Util.h"

using namespace std;

int main()
{
	Tensor<double> data = getCSVDataSet("C:\\Users\\jungh\\OneDrive\\πŸ≈¡ »≠∏È\\iris_one_hot_encoded.csv", false);
	Tensor<double> input, target;
	parseInputAndTarget(data, input, target, 4, 3);
	NeuralNet net(1000, 3, OPT_OPTM::ADAM, OPT_LOSF::CROSSENTROPY, OPT_INIT::RANDOM, OPT_NORM::USENORM, 0.f);
	net.addLayer(OPT_LYR::INPUT, 4);
	net.addLayer(OPT_LYR::FULLYCONNECTED, 5, OPT_ACTF::RELU);
	net.addLayer(OPT_LYR::FULLYCONNECTED, 3, OPT_ACTF::SOFTMAX);
	net.setInput(input);
	net.setTarget(target);
	net.train();
	system("pause");

	return 0;
}