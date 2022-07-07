#pragma once
#include <string>
#include <fstream>

#include "Tensor.h"

Tensor<double> getCSVDataSet(const char* path, bool isHead)
{
	Tensor<double> data;
	std::fstream CSVFile;
	CSVFile.open(path, std::ios::in);
	bool fileOpen = CSVFile.is_open();
	if (!fileOpen)
	{
		printf("[Error]File Not Found");
	}
	else
	{
		char charLine[4096];
		std::string strLine = "";
		int Nrow = 0;
		int Ncol = 0;
		while (CSVFile.getline(charLine, 4096))
		{
			if (!isHead)
			{
				strLine += charLine;
				strLine += ",";
				Nrow++;
				if (Ncol == 0)
				{
					for (int i = 0; i < strLine.length(); i++)
					{
						if (strLine.find(",") == -1)break;
						else if (strLine.find(",", i) <= i)Ncol++;
					}
				}
			}
			else
			{
				isHead = false;
			}
		}
		data.resize(Nrow, Ncol);

		int prevComma = 0;
		int postComma = 0;
		for (int ridx = 0; ridx < Nrow; ridx++)
		{
			for (int cidx = 0; cidx < Ncol; cidx++)
			{
				postComma = strLine.find(",", prevComma);
				std::string strData = strLine.substr(prevComma, postComma - prevComma);
				data(ridx,cidx) = std::stod(strData);
				prevComma = postComma + 1;
			}
		}
		CSVFile.close();
	}
	return data;
}

void parseInputAndTarget(const Tensor<double>& data_, Tensor<double>& input_, Tensor<double>& target_, const int input_num_, const int target_num_)
{
	int dataNum = data_.dim(0);
	int fieldNum = data_.dim(1);
	input_.resize(dataNum, input_num_);
	target_.resize(dataNum, target_num_);

	for (int i = 0; i < dataNum; i++)
	{
		for (int j = 0; j < fieldNum; j++)
		{
			if (j < input_num_)
			{
				input_(i, j) = data_.element(i, j);
			}
			else if (j < input_num_ + target_num_)
			{
				target_(i, j - input_num_) = data_.element(i, j);
			}
		}
	}
}