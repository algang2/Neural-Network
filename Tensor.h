#pragma once
template <typename T = double>
class Tensor
{
public:
	Tensor();
	Tensor(int dim_0_);
	Tensor(int dim_0_, int dim_1_);
	Tensor(int dim_0_, int dim_1_, int dim_2_);
	Tensor(int dim_0_, int dim_1_, int dim_2_, int dim_3_);
	Tensor(const Tensor& rhs_);
	~Tensor();
public:
	T& operator()(int dim_0_);
	T& operator()(int dim_0_, int dim_1_);
	T& operator()(int dim_0_, int dim_1_, int dim_2_);
	T& operator()(int dim_0_, int dim_1_, int dim_2_, int dim_3_);
	Tensor& operator=(const T& rhs_);
	Tensor& operator=(const Tensor& rhs_);
	Tensor& operator+=(const Tensor& rhs_);
	Tensor& operator-=(const Tensor& rhs_);
	Tensor& operator*=(const Tensor& rhs_);
	Tensor& operator/=(const Tensor& rhs_);
	Tensor operator+(const Tensor& rhs_);
	Tensor operator-(const Tensor& rhs_);
	Tensor operator*(const Tensor& rhs_);
	Tensor operator/(const Tensor& rhs_);
public:
	int size() const;
	int dim(int dim_) const;
	void resize(int dim_0_);
	void resize(int dim_0_, int dim_1_);
	void resize(int dim_0_, int dim_1_, int dim_2_);
	void resize(int dim_0_, int dim_1_, int dim_2_, int dim_3_);
	T element(int dim_0_) const;
	T element(int dim_0_, int dim_1_) const;
	T element(int dim_0_, int dim_1_, int dim_2_) const;
	T element(int dim_0_, int dim_1_, int dim_2_, int dim_3_) const;
	void erase();
private:
	int dim_0, dim_1, dim_2, dim_3;
	T* tensor;
};

template <typename T>
Tensor<T>::Tensor() :dim_0(0), dim_1(0), dim_2(0), dim_3(0)
{
	tensor = nullptr;
}

template <typename T>
Tensor<T>::Tensor(int dim_0_) :dim_0(dim_0_), dim_1(1), dim_2(1), dim_3(1)
{
	tensor = new T[dim_0 * dim_1 * dim_2 * dim_3]();
}

template <typename T>
Tensor<T>::Tensor(int dim_0_, int dim_1_) :dim_0(dim_0_), dim_1(dim_1_), dim_2(1), dim_3(1)
{
	tensor = new T[dim_0 * dim_1 * dim_2 * dim_3]();
}

template <typename T>
Tensor<T>::Tensor(int dim_0_, int dim_1_, int dim_2_) :dim_0(dim_0_), dim_1(dim_1_), dim_2(dim_2_), dim_3(1)
{
	tensor = new T[dim_0 * dim_1 * dim_2 * dim_3]();
}

template <typename T>
Tensor<T>::Tensor(int dim_0_, int dim_1_, int dim_2_, int dim_3_) :dim_0(dim_0_), dim_1(dim_1_), dim_2(dim_2_), dim_3(dim_3_)
{
	tensor = new T[dim_0 * dim_1 * dim_2 * dim_3]();
}

template <typename T>
Tensor<T>::Tensor(const Tensor& rhs_)
{
	dim_0 = rhs_.dim_0; dim_1 = rhs_.dim_1; dim_2 = rhs_.dim_2; dim_3 = rhs_.dim_3;
	int size = rhs_.size();
	tensor = new T[size];
	for (int idx = 0; idx < size; idx++)tensor[idx] = rhs_.tensor[idx];
}

template <typename T>
Tensor<T>::~Tensor()
{
	delete[] tensor;
}

template <typename T>
T& Tensor<T>::operator()(int dim_0_)
{
	return tensor[dim_0_];
}

template <typename T>
T& Tensor<T>::operator()(int dim_0_, int dim_1_)
{
	return tensor[dim_0_ * dim_1 + dim_1_];
}

template <typename T>
T& Tensor<T>::operator()(int dim_0_, int dim_1_, int dim_2_)
{
	return tensor[dim_0_ * dim_1 * dim_2 + dim_1_ * dim_2 + dim_2_];
}

template <typename T>
T& Tensor<T>::operator()(int dim_0_, int dim_1_, int dim_2_, int dim_3_)
{
	return tensor[dim_0_ * dim_1 * dim_2 * dim_3 + dim_1_ * dim_2 * dim_3 + dim_2_ * dim_3 + dim_3_];
}

template <typename T>
Tensor<T>& Tensor<T>::operator=(const T& rhs_)
{
	int size = this->size();
	for (int idx = 0; idx < size; idx++)tensor[idx] = rhs_;
	return *this;
}

template <typename T>
Tensor<T>& Tensor<T>::operator=(const Tensor& rhs_)
{
	dim_0 = rhs_.dim_0; dim_1 = rhs_.dim_1; dim_2 = rhs_.dim_2; dim_3 = rhs_.dim_3;
	int size = rhs_.size();
	if (tensor != nullptr)delete[] tensor;
	tensor = new T[size];
	for (int idx = 0; idx < size; idx++)tensor[idx] = rhs_.tensor[idx];
	return *this;
}

template <typename T>
Tensor<T>& Tensor<T>::operator+=(const Tensor& rhs_)
{
	int size = rhs_.size();
	for (int idx = 0; idx < size; idx++)tensor[idx] += rhs_.tensor[idx];
	return *this;
}

template <typename T>
Tensor<T>& Tensor<T>::operator-=(const Tensor& rhs_)
{
	int size = rhs_.size();
	for (int idx = 0; idx < size; idx++)tensor[idx] -= rhs_.tensor[idx];
	return *this;
}

template <typename T>
Tensor<T>& Tensor<T>::operator*=(const Tensor& rhs_)
{
	int size = rhs_.size();
	for (int idx = 0; idx < size; idx++)tensor[idx] *= rhs_.tensor[idx];
	return *this;
}

template <typename T>
Tensor<T>& Tensor<T>::operator/=(const Tensor& rhs_)
{
	int size = rhs_.size();
	for (int idx = 0; idx < size; idx++)tensor[idx] /= rhs_.tensor[idx];
	return *this;
}

template <typename T>
Tensor<T> Tensor<T>::operator+(const Tensor& rhs_)
{
	Tensor tmp(dim_0, dim_1, dim_2, dim_3);
	int size = rhs_.size();
	for (int idx = 0; idx < size; idx++)tmp.tensor[idx] = tensor[idx] + rhs_.tensor[idx];
	return Tensor(tmp);
}

template <typename T>
Tensor<T> Tensor<T>::operator-(const Tensor& rhs_)
{
	Tensor tmp(dim_0, dim_1, dim_2, dim_3);
	int size = rhs_.size();
	for (int idx = 0; idx < size; idx++)tmp.tensor[idx] = tensor[idx] - rhs_.tensor[idx];
	return Tensor(tmp);
}

template <typename T>
Tensor<T> Tensor<T>::operator*(const Tensor& rhs_)
{
	Tensor tmp(dim_0, dim_1, dim_2, dim_3);
	int size = rhs_.size();
	for (int idx = 0; idx < size; idx++)tmp.tensor[idx] = tensor[idx] * rhs_.tensor[idx];
	return Tensor(tmp);
}

template <typename T>
Tensor<T> Tensor<T>::operator/(const Tensor& rhs_)
{
	Tensor tmp(dim_0, dim_1, dim_2, dim_3);
	int size = rhs_.size();
	for (int idx = 0; idx < size; idx++)tmp.tensor[idx] = tensor[idx] / rhs_.tensor[idx];
	return Tensor(tmp);
}

template <typename T>
Tensor<T> operator+(const double& val_, const Tensor<T>& rhs_)
{
	Tensor<T> tmp(rhs_);
	int size = rhs_.size();
	for (int idx = 0; idx < size; idx++)tmp(idx) = val_ + rhs_.element(idx);
	return Tensor<T>(tmp);
}

template <typename T>
Tensor<T> operator+(const Tensor<T>& rhs_, const double& val_)
{
	Tensor<T> tmp(rhs_);
	int size = rhs_.size();
	for (int idx = 0; idx < size; idx++)tmp(idx) = rhs_.element(idx) + val_;
	return Tensor<T>(tmp);
}

template <typename T>
Tensor<T> operator-(const double& val_, const Tensor<T>& rhs_)
{
	Tensor<T> tmp(rhs_);
	int size = rhs_.size();
	for (int idx = 0; idx < size; idx++)tmp(idx) = val_ - rhs_.element(idx);
	return Tensor<T>(tmp);
}

template <typename T>
Tensor<T> operator-(const Tensor<T>& rhs_, const double& val_)
{
	Tensor<T> tmp(rhs_);
	int size = rhs_.size();
	for (int idx = 0; idx < size; idx++)tmp(idx) = rhs_.element(idx) - val_;
	return Tensor<T>(tmp);
}

template <typename T>
Tensor<T> operator*(const double& val_, const Tensor<T>& rhs_)
{
	Tensor<T> tmp(rhs_);
	int size = rhs_.size();
	for (int idx = 0; idx < size; idx++)tmp(idx) = val_ * rhs_.element(idx);
	return Tensor<T>(tmp);
}

template <typename T>
Tensor<T> operator*(const Tensor<T>& rhs_, const double& val_)
{
	Tensor<T> tmp(rhs_);
	int size = rhs_.size();
	for (int idx = 0; idx < size; idx++)tmp(idx) = rhs_.element(idx) * val_;
	return Tensor<T>(tmp);
}

template <typename T>
Tensor<T> operator/(const double& val_, const Tensor<T>& rhs_)
{
	Tensor<T> tmp(rhs_);
	int size = rhs_.size();
	for (int idx = 0; idx < size; idx++)tmp(idx) = val_ / rhs_.element(idx);
	return Tensor<T>(tmp);
}

template <typename T>
Tensor<T> operator/(const Tensor<T>& rhs_, const double& val_)
{
	Tensor<T> tmp(rhs_);
	int size = rhs_.size();
	for (int idx = 0; idx < size; idx++)tmp(idx) = rhs_.element(idx) / val_;
	return Tensor<T>(tmp);
}

template <typename T>
int Tensor<T>::size() const
{
	return dim_0 * dim_1 * dim_2 * dim_3;
}

template <typename T>
int Tensor<T>::dim(int dim_) const
{
	if (dim_ == 0)return dim_0;
	else if (dim_ == 1)return dim_1;
	else if (dim_ == 2)return dim_2;
	else if (dim_ == 3)return dim_3;
	else return 0;
}

template <typename T>
void Tensor<T>::resize(int dim_0_)
{
	if (tensor != nullptr)delete[] tensor;
	dim_0 = dim_0_; dim_1 = 1; dim_2 = 1; dim_3 = 1;
	tensor = new T[dim_0 * dim_1 * dim_2 * dim_3]();
}

template <typename T>
void Tensor<T>::resize(int dim_0_, int dim_1_)
{
	if (tensor != nullptr)delete[] tensor;
	dim_0 = dim_0_; dim_1 = dim_1_; dim_2 = 1; dim_3 = 1;
	tensor = new T[dim_0 * dim_1 * dim_2 * dim_3]();
}

template <typename T>
void Tensor<T>::resize(int dim_0_, int dim_1_, int dim_2_)
{
	if (tensor != nullptr)delete[] tensor;
	dim_0 = dim_0_; dim_1 = dim_1_; dim_2 = dim_2_; dim_3 = 1;
	tensor = new T[dim_0 * dim_1 * dim_2 * dim_3]();
}

template <typename T>
void Tensor<T>::resize(int dim_0_, int dim_1_, int dim_2_, int dim_3_)
{
	if (tensor != nullptr)delete[] tensor;
	dim_0 = dim_0_; dim_1 = dim_1_; dim_2 = dim_2_; dim_3 = dim_3_;
	tensor = new T[dim_0 * dim_1 * dim_2 * dim_3]();
}

template <typename T>
T Tensor<T>::element(int dim_0_) const
{
	return tensor[dim_0_];
}

template <typename T>
T Tensor<T>::element(int dim_0_, int dim_1_) const
{
	return tensor[dim_0_ * dim_1 + dim_1_];
}

template <typename T>
T Tensor<T>::element(int dim_0_, int dim_1_, int dim_2_) const
{
	return tensor[dim_0_ * dim_1 * dim_2 + dim_1_ * dim_2 + dim_2_];
}

template <typename T>
T Tensor<T>::element(int dim_0_, int dim_1_, int dim_2_, int dim_3_) const
{
	return tensor[dim_0_ * dim_1 * dim_2 * dim_3 + dim_1_ * dim_2 * dim_3 + dim_2_ * dim_3 + dim_3_];
}

template <typename T>
void Tensor<T>::erase()
{
	for (int i = 0; i < size(); i++)tensor[i] = 0;
}