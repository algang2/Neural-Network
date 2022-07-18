#pragma once
#include <string>
#include <fstream>

class BinaryReader
{
public:
	BinaryReader();
	BinaryReader(const std::string& dir_);
	~BinaryReader();
	void setNext(const std::string& binary_, bool new_ = false);
	std::string getNext();
	void setBinary(const std::string& binary_);
	std::string getBinary();
	void save(const std::string& dir_);
private:
	std::string binary;
	int pos;
};