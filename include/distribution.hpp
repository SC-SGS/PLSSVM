#pragma once
#include <vector>
#include <numeric>
#include <functional>
#include "typedef.hpp"

class distribution{
    public:
    distribution(const size_t number_devices, const size_t number_lines, const std::vector<real_t> parameter){
        if(number_devices != parameter.size()) throw "Distribution error!";
        real_t sum = std::accumulate(parameter.begin(), parameter.end(), 0);
        for(real_t value :parameter){
            distr.push_back(value / sum * number_lines);
        }
        distr[number_devices-1] += std::accumulate(distr.begin(), distr.end(), number_lines, std::minus<int>());
    }

//TODO: optional
    // distribution(const size_t number_devices, const size_t number_lines){
        
    // }
    
    bool isValid(size_t number_devices, std::vector<real_t> parameter){
        return (number_devices <= parameter.size() && std::accumulate(parameter.begin(), parameter.end(), 0) <= 1.0);
    }

    std::vector<int>distr;


};
