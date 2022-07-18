#include "BinaryReader.h"

BinaryReader::BinaryReader()
{
	pos = 0;
}

BinaryReader::BinaryReader(const std::string& dir_)
{
	std::ifstream fin(dir_.c_str(), std::ifstream::binary);
	if (fin) {
		fin.seekg(0, fin.end);
		int len = (int)fin.tellg();
		fin.seekg(0, fin.beg);
		char* buffer = new char[len];
		fin.read(buffer, len);
		fin.close();
		std::string tmp(buffer);
		binary = tmp;
		delete[] buffer;
	}
	pos = 0;
}

BinaryReader::~BinaryReader()
{

}

void BinaryReader::setNext(const std::string& binary_, bool new_)
{
	if (new_)
	{
		binary = binary_;
	}
	else
	{
		binary += "," + binary_;
	}
}

std::string BinaryReader::getNext()
{
	std::string ret;
	while (binary[pos] != ',')
	{
		ret += binary[pos];
		pos++;
		if (pos == binary.length())
		{
			return ret;
		}
	}
	pos++;
	return ret;
}

void BinaryReader::setBinary(const std::string& binary_)
{
	binary = binary_;
}

std::string BinaryReader::getBinary()
{
	return binary;
}

void BinaryReader::save(const std::string& dir_)
{
	std::ofstream fout;
	fout.open(dir_.c_str(), std::ios::out | std::ios::binary);
	int len = binary.length();
	if (fout.is_open()) 
	{
		fout.write((const char*)binary.c_str(), len);
		fout.close();
	}
}