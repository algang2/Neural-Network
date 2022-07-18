#pragma once
#include <string>
#include <fstream>

#include "Tensor.h"

Tensor<std::string> getCSVDataSet(const char* path, bool isHead)
{
	Tensor<std::string> data;
	std::fstream CSVFile;
	CSVFile.open(path, std::ios::in);
	if (!CSVFile.is_open())
	{
		printf("[Error]File Not Found.\n");
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
				data(ridx,cidx) = strLine.substr(prevComma, postComma - prevComma);
				prevComma = postComma + 1;
			}
		}
		CSVFile.close();
	}
	return data;
}

void parseInputAndTarget(const Tensor<std::string>& data_, Tensor<std::string>& input_, Tensor<std::string>& target_, const int target_pos_)
{
	int dataNum = data_.dim(0);
	int fieldNum = data_.dim(1);
	input_.resize(dataNum, fieldNum - 1);
	target_.resize(dataNum, 1);

	for (int i = 0; i < dataNum; i++)
	{
		for (int j = 0, input_dim_1 = 0; j < fieldNum; j++)
		{
			if (j != target_pos_)
			{
				input_(i, input_dim_1) = data_.element(i, j);
				input_dim_1++;
			}
			else
			{
				target_(i, 0) = data_.element(i, j);
			}
		}
	}
}

void printResult(const Tensor<std::string>& output_, const Tensor<std::string>& target_)
{
	if (target_.size() == 0)
	{
		return;
	}
	int dim_0 = output_.dim(0);
	for (int i = 0; i < dim_0; i++)
	{
		printf("%s[%s]\n", output_.element(i).c_str(), target_.element(i).c_str());
	}
}