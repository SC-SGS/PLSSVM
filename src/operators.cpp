#include "operators.hpp"
#include <algorithm>
#include <numeric>
#include <functional>


std::vector<double> operator - (const std::vector<double> &vec1,const std::vector<double> &vec2)
{
    std::vector<double> result(vec1.size());
    for(unsigned i  = 0; i < vec1.size(); ++i){
        result[i] = (vec1[i] - vec2[i]);
    }
    return result;
}

std::vector<double> operator + (const std::vector<double> &vec1,const std::vector<double> &vec2)
{
	std::vector<double> result(vec1.size());
    for(unsigned i  = 0; i < vec1.size(); ++i){
		result[i] = (vec1[i] + vec2[i]);
    }
    return result;

}

std::vector<double>& operator += (std::vector<double> &result,const std::vector<double> &vec2)
{
    for(unsigned i  = 0; i < result.size(); ++i){
        result[i] += vec2[i];
    }
    return result;
}

double* operator += (double* result,const std::vector<double> &vec2)
{
    for(unsigned i  = 0; i < vec2.size(); ++i){
        result[i] += vec2[i];
    }
    return result;
}

std::vector<double> operator *(const std::vector<std::vector<double> > &matr, const std::vector<double> &vec)
{
    std::vector<double> result(matr.size(),0.0);
    for(unsigned i = 0; i < matr.size(); ++i){
        for(unsigned j = 0; j < vec.size(); ++j){
            result[i] += matr[i][j] * vec[j] ;
        }
    }
    return result;
}


double operator *(const std::vector<double> &vec1,const std::vector<double> &vec2)
{
    double result = 0.0;
    for(unsigned i = 0; i < vec1.size(); ++i){
        result += vec1[i] * vec2[i];
    }
	return result;
}

double operator *(double* vec1,const std::vector<double> &vec2)
{
    double result = 0.0;
    for(unsigned i = 0; i < vec2.size(); ++i){
        result += vec1[i] * vec2[i];
    }
	return result;
}

std::ostream& operator<<(std::ostream &out,const std::vector<std::vector<double> > &matr)
{
    for(unsigned i = 0; i < matr.size(); ++i){
        out << matr[i] << std::endl;
    }
    return out;
}

std::ostream& operator<<(std::ostream &out, const std::vector<double> &vec)
{
    for(unsigned i = 0; i < vec.size() ; ++i){
        if(vec[i] != 0 ) out << i << ":" << vec[i] << " ";
    }
    return out;
}

std::vector<double> operator * (const double& value, std::vector<double> vec)
{
    for(unsigned i = 0; i < vec.size(); ++i ){
        vec[i] *= value;
    }
    return vec;
}

double operator * (const std::vector<double>&vec1,double* vec2)
{
	double result = 0.0;
    for(unsigned i = 0; i < vec1.size(); ++i ){
        result += vec1[i] * vec2[i];
    }
    return result;
}

std::vector<double> operator * (const std::vector<double>& vec, const double& value)
{
    return value * vec;
}

std::vector<std::vector<double> > dot(const std::vector<double>& vec1, const std::vector<double>& vec2)
{
	std::vector<std::vector<double> > result(vec1.size(),std::vector<double>(vec1.size(),0));
	for(unsigned i = 0; i < vec1.size(); ++i)
    {
        for(unsigned j = 0; j < vec1.size(); ++j)
        {
            result[i][j] += vec1[i] * vec2[j];
        }
    }
    return result;
}

std::vector<std::vector<double> >& operator -= (std::vector<std::vector<double>> &result, const std::vector<std::vector<double> > &matr)
{
    for(unsigned i = 0; i < result.size(); ++i)
    {
        for(unsigned j = 0; j < result[0].size(); ++j)
        {
            result[i][j] -= matr[i][j];
        }
    }
    return result;
}

std::vector<std::vector<double> >& operator +=(std::vector<std::vector<double> > &result, const double &value)
{
    for(unsigned i = 0; i < result.size(); ++i)
    {
        for(unsigned j = 0; j < result[0].size(); ++j)
        {
            result[i][j] += value;
        }
    }
    return result;
}
std::vector<double>& operator -= (std::vector<double> &result, const double &value)
{
    for(unsigned i = 0; i < result.size(); ++i){
        result[i] -= value;
    }
    return result;
};


double sum(std::vector<double> & vec)
{
    double result = 0;
	for(unsigned i = 0; i < vec.size(); ++i){
        result += vec[i];
    }
    return result;
};

double mult(double* vec1,double* vec2,int dim){
	double result = 0.0;
    for(unsigned i = 0; i < dim; ++i){
        result += vec1[i] * vec2[i];
    }
	return result;
};

double* mult(double* vec, double val, int dim){
	return mult(val,vec, dim);
};

double* mult (double value, double* vec, int dim){
	for(int i = 0; i < dim; ++i){
		vec[i] *= value;
	}
	return vec;
};

double* add (double* vec, double value, int dim){
	return add(value, vec, dim);
};

double* add (double value, double* vec, int dim){
	for(unsigned i = 0; i < dim; ++i){
		vec[i] += value;
	}
	return vec;
};

double* add (double* vec1, double* vec2, int dim){
	double* result =  new double[dim];
	for(unsigned i = 0; i < dim; ++i){
		result[i] = vec1[i] + vec2[i];
	}
	return result;
};

double* add (double* vec1, double* vec2, double* result, int dim){
	for(unsigned i = 0; i < dim; ++i){
		result[i] = vec1[i] + vec2[i];
	}
	return result;
};


std::vector<double>& operator += (std::vector<double>&vec1, double* vec2){
	for(int i = 0; i < vec1.size(); ++i){
		vec1[i] += vec2[i];
	}
	return vec1;
};


bool endsWith(std::string mainStr, std::string toMatch)
{
	auto it = toMatch.begin();
	return mainStr.size() >= toMatch.size() &&
			std::all_of(std::next(mainStr.begin(),mainStr.size() - toMatch.size()), mainStr.end(), [&it](const char & c){
				return ::tolower(c) == ::tolower(*(it++))  ;
	} );
}