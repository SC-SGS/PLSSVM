#ifndef OPERATORS_HPP_INCLUDED
#define OPERATORS_HPP_INCLUDED

#include <vector>
#include <iostream>

using real_t = float;

void mult_vec_vec(real_t *vec1, real_t *vec2, int dim, real_t *result);
//void mult_vec_scal(real_t *vec1, real_t scal, int dim, real_t *result);
void minus(real_t *vec1, real_t *vec2, int dim, real_t *result);
void add(real_t *vec1, real_t *vec2, int dim, real_t *result);
void add_inplace(real_t *vec1, int dim, real_t *result);


std::vector<real_t> operator - (const std::vector<real_t>&, const std::vector<real_t>&);
std::vector<real_t> operator + (const std::vector<real_t>&, const std::vector<real_t>&);
std::vector<real_t>& operator += (std::vector<real_t>&, const std::vector<real_t>&);
std::vector<real_t>& operator += (std::vector<real_t>&, real_t*);
std::vector<real_t>& operator -= (std::vector<real_t>&, const real_t&);
std::vector<std::vector<real_t> >& operator -= (std::vector<std::vector<real_t> >&, const std::vector<std::vector<real_t> >&);
std::vector<std::vector<real_t> >& operator += (std::vector<std::vector<real_t> >&, const real_t&);
std::vector<real_t> operator * (const std::vector<std::vector<real_t> >&, const std::vector<real_t>&);
std::vector<real_t> operator * (const real_t&, std::vector<real_t>);
std::vector<real_t> operator * (const std::vector<real_t>&, const real_t&);
std::vector<std::vector<real_t> > dot (const std::vector<real_t>&,const std::vector<real_t>&);
real_t operator * (const std::vector<real_t>&,const std::vector<real_t>&);
std::ostream& operator<<(std::ostream&, const std::vector<std::vector<real_t> >&);
std::ostream& operator<<(std::ostream&, const std::vector<real_t>&);
real_t sum(std::vector<real_t>& );

real_t mult(real_t*, real_t*, int );
real_t* mult(real_t,real_t*, int);
real_t* mult(real_t* vec, real_t val, int);
real_t* add (real_t, real_t*, int);
real_t* add (real_t* vec, real_t value, int);
real_t* add (real_t*, real_t*, int);
real_t* add (real_t*, real_t*, real_t*, int);

bool endsWith(std::string, std::string);

#endif // OPERATORS_HPP_INCLUDED
