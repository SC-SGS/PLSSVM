#ifndef OPERATORS_HPP_INCLUDED
#define OPERATORS_HPP_INCLUDED

#include <vector>
#include <iostream>

void mult_vec_vec(double *vec1, double *vec2, int dim, double *result);
//void mult_vec_scal(double *vec1, double scal, int dim, double *result);
void minus(double *vec1, double *vec2, int dim, double *result);
void add(double *vec1, double *vec2, int dim, double *result);
void add_inplace(double *vec1, int dim, double *result);


std::vector<double> operator - (const std::vector<double>&, const std::vector<double>&);
std::vector<double> operator + (const std::vector<double>&, const std::vector<double>&);
std::vector<double>& operator += (std::vector<double>&, const std::vector<double>&);
std::vector<double>& operator += (std::vector<double>&, double*);
std::vector<double>& operator -= (std::vector<double>&, const double&);
std::vector<std::vector<double> >& operator -= (std::vector<std::vector<double> >&, const std::vector<std::vector<double> >&);
std::vector<std::vector<double> >& operator += (std::vector<std::vector<double> >&, const double&);
std::vector<double> operator * (const std::vector<std::vector<double> >&, const std::vector<double>&);
std::vector<double> operator * (const double&, std::vector<double>);
std::vector<double> operator * (const std::vector<double>&, const double&);
std::vector<std::vector<double> > dot (const std::vector<double>&,const std::vector<double>&);
double operator * (const std::vector<double>&,const std::vector<double>&);
std::ostream& operator<<(std::ostream&, const std::vector<std::vector<double> >&);
std::ostream& operator<<(std::ostream&, const std::vector<double>&);
double sum(std::vector<double>& );

double mult(double*, double*, int );
double* mult(double,double*, int);
double* mult(double* vec, double val, int);
double* add (double, double*, int);
double* add (double* vec, double value, int);
double* add (double*, double*, int);
double* add (double*, double*, double*, int);

bool endsWith(std::string, std::string);

#endif // OPERATORS_HPP_INCLUDED
