// //#include <plssvm/operators.hpp>
// #include <vector>
// #include <iostream>

// #include <plssvm/typedef.hpp>
// #include <algorithm>
// #include <numeric>
// #include <functional>

// std::vector<real_t> operator - (const std::vector<real_t> &vec1,const std::vector<real_t> &vec2)
// {
//     std::vector<real_t> result(vec1.size());
//     for(unsigned i  = 0; i < vec1.size(); ++i){
//         result[i] = (vec1[i] - vec2[i]);
//     }
//     return result;
// }

// std::vector<real_t> operator + (const std::vector<real_t> &vec1,const std::vector<real_t> &vec2)
// {
// 	std::vector<real_t> result(vec1.size());
//     for(unsigned i  = 0; i < vec1.size(); ++i){
// 		result[i] = (vec1[i] + vec2[i]);
//     }
//     return result;

// }

// std::vector<real_t>& operator += (std::vector<real_t> &result,const std::vector<real_t> &vec2)
// {
//     for(unsigned i  = 0; i < result.size(); ++i){
//         result[i] += vec2[i];
//     }
//     return result;
// }

// real_t* operator += (real_t* result,const std::vector<real_t> &vec2)
// {
//     for(unsigned i  = 0; i < vec2.size(); ++i){
//         result[i] += vec2[i];
//     }
//     return result;
// }

// std::vector<real_t> operator *(const std::vector<std::vector<real_t> > &matr, const std::vector<real_t> &vec)
// {
//     std::vector<real_t> result(matr.size(),0.0);
//     for(unsigned i = 0; i < matr.size(); ++i){
//         for(unsigned j = 0; j < vec.size(); ++j){
//             result[i] += matr[i][j] * vec[j] ;
//         }
//     }
//     return result;
// }

// real_t operator *(const std::vector<real_t> &vec1,const std::vector<real_t> &vec2)
// {
//     real_t result = 0.0;
//     for(unsigned i = 0; i < vec1.size(); ++i){
//         result += vec1[i] * vec2[i];
//     }
// 	return result;
// }

// real_t operator *(real_t* vec1,const std::vector<real_t> &vec2)
// {
//     real_t result = 0.0;
//     for(unsigned i = 0; i < vec2.size(); ++i){
//         result += vec1[i] * vec2[i];
//     }
// 	return result;
// }

// std::ostream& operator<<(std::ostream &out, const std::vector<real_t> &vec)
// {
//     char buffer[20];
//     for(unsigned i = 0; i < vec.size() ; ++i){
//         if(vec[i] != 0 ){
//             sprintf(buffer, "%i:%e ",i,vec[i]);
//             out << buffer;
//         } //out << i << ":" << vec[i] << " ";
//     }
//     return out;
// }
// std::ostream& operator<<(std::ostream &out,const std::vector<std::vector<real_t> > &matr)
// {
//     for(unsigned i = 0; i < matr.size(); ++i){
//         out << matr[i] << '\n';
//     }
//     return out;
// }

// std::vector<real_t> operator * (const real_t& value, std::vector<real_t> vec)
// {
//     for(unsigned i = 0; i < vec.size(); ++i ){
//         vec[i] *= value;
//     }
//     return vec;
// }

// real_t operator * (const std::vector<real_t>&vec1,real_t* vec2)
// {
// 	real_t result = 0.0;
//     for(unsigned i = 0; i < vec1.size(); ++i ){
//         result += vec1[i] * vec2[i];
//     }
//     return result;
// }

// std::vector<real_t> operator * (const std::vector<real_t>& vec, const real_t& value)
// {
//     return value * vec;
// }

// std::vector<std::vector<real_t> > dot(const std::vector<real_t>& vec1, const std::vector<real_t>& vec2)
// {
// 	std::vector<std::vector<real_t> > result(vec1.size(),std::vector<real_t>(vec1.size(),0));
// 	for(unsigned i = 0; i < vec1.size(); ++i)
//     {
//         for(unsigned j = 0; j < vec1.size(); ++j)
//         {
//             result[i][j] += vec1[i] * vec2[j];
//         }
//     }
//     return result;
// }

// std::vector<std::vector<real_t> >& operator -= (std::vector<std::vector<real_t>> &result, const std::vector<std::vector<real_t> > &matr)
// {
//     for(unsigned i = 0; i < result.size(); ++i)
//     {
//         for(unsigned j = 0; j < result[0].size(); ++j)
//         {
//             result[i][j] -= matr[i][j];
//         }
//     }
//     return result;
// }

// std::vector<std::vector<real_t> >& operator +=(std::vector<std::vector<real_t> > &result, const real_t &value)
// {
//     for(unsigned i = 0; i < result.size(); ++i)
//     {
//         for(unsigned j = 0; j < result[0].size(); ++j)
//         {
//             result[i][j] += value;
//         }
//     }
//     return result;
// }
// std::vector<real_t>& operator -= (std::vector<real_t> &result, const real_t &value)
// {
//     for(unsigned i = 0; i < result.size(); ++i){
//         result[i] -= value;
//     }
//     return result;
// };

// real_t sum(std::vector<real_t> & vec)
// {
//     real_t result = 0;
// 	for(unsigned i = 0; i < vec.size(); ++i){
//         result += vec[i];
//     }
//     return result;
// };

// real_t mult(real_t* vec1,real_t* vec2,int dim){
// 	real_t result = 0.0;
//     for(unsigned i = 0; i < dim; ++i){
//         result += vec1[i] * vec2[i];
//     }
// 	return result;
// };

// real_t* mult (real_t value, real_t* vec, int dim){
// 	for(int i = 0; i < dim; ++i){
// 		vec[i] *= value;
// 	}
// 	return vec;
// };

// real_t* mult(real_t* vec, real_t val, int dim){
// 	return mult(val,vec, dim);
// };

// real_t* add (real_t value, real_t* vec, int dim){
// 	for(unsigned i = 0; i < dim; ++i){
// 		vec[i] += value;
// 	}
// 	return vec;
// };

// real_t* add (real_t* vec1, real_t* vec2, int dim){
// 	real_t* result =  new real_t[dim];
// 	for(unsigned i = 0; i < dim; ++i){
// 		result[i] = vec1[i] + vec2[i];
// 	}
// 	return result;
// };

// real_t* add (real_t* vec1, real_t* vec2, real_t* result, int dim){
// 	for(unsigned i = 0; i < dim; ++i){
// 		result[i] = vec1[i] + vec2[i];
// 	}
// 	return result;
// };

// real_t* add (real_t* vec, real_t value, int dim){
// 	return add(value, vec, dim);
// };

// std::vector<real_t>& operator += (std::vector<real_t>&vec1, real_t* vec2){
// 	for(int i = 0; i < vec1.size(); ++i){
// 		vec1[i] += vec2[i];
// 	}
// 	return vec1;
// };

// bool endsWith(std::string mainStr, std::string toMatch)
// {
// 	auto it = toMatch.begin();
// 	return mainStr.size() >= toMatch.size() &&
// 			std::all_of(std::next(mainStr.begin(),mainStr.size() - toMatch.size()), mainStr.end(), [&it](const char & c){
// 				return ::tolower(c) == ::tolower(*(it++))  ;
// 	} );
// }