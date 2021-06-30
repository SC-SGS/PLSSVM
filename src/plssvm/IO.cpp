#include <plssvm/CSVM.hpp>
#include <plssvm/operators.hpp>
#include <plssvm/exceptions.hpp>
#include <plssvm/string_utility.hpp>

#include <fmt/core.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include <fast_float/fast_float.h>

namespace plssvm {

//read libsvm file
void CSVM::libsvmParser(const std::string_view filename) {
    std::vector<std::string> data_lines;

    {
        std::ifstream file{filename.data()};
        if (file.fail()) {
            throw file_not_found_exception{fmt::format("Couldn't find file: '{}'!", filename)};
        }
        std::string line;
        while (std::getline(file, line)) {
            std::string_view trimmed = util::trim_left(line);
            if (!trimmed.empty() && !util::starts_with(trimmed, '#')) {
                data_lines.push_back(std::move(line));
            }
        }
    }

    value.resize(data_lines.size());
    data.resize(data_lines.size());

    std::size_t max_size = 0;

    #pragma omp parallel for reduction(max:max_size)
    for (std::size_t i = 0; i < data.size(); ++i) {
      std::string_view line = data_lines[i];

      // get class
      std::size_t pos = line.find_first_of(' ');
      value[i] = util::convert_to<real_t, invalid_file_format_exception>(line.substr(0, pos)) > real_t{0.0} ? 1 : -1;
      // value[i] = std::copysign(1.0, util::convert_to<real_t>(line.substr(0, pos)));

      // get data
      std::vector<real_t> vline(max_size);
      std::size_t next_pos = line.find_first_of(':', pos);
      while (next_pos != std::string_view::npos) {
        // get index
        const auto index = util::convert_to<unsigned long, invalid_file_format_exception>(line.substr(pos, next_pos - pos));
        if (index >= vline.size()) {
          vline.resize(index + 1);
        }
        pos = next_pos + 1;

        // get value
        next_pos = line.find_first_of(',', pos);
        vline[index] = util::convert_to<real_t, invalid_file_format_exception>(line.substr(pos, next_pos - pos));

        if (next_pos == std::string_view::npos) {
          break;
        }
        pos = next_pos + 1;
        next_pos = line.find_first_of(':', pos);
      }
      max_size = std::max(max_size, vline.size());
      data[i] = std::move(vline);
    }

    #pragma omp parallel for
    for (std::size_t i = 0; i < data.size(); ++i) {
      data[i].resize(max_size);
    }

    // update values
    num_data_points = data.size();
    num_features = max_size;

    // no features were parsed -> invalid file
    if (num_features == 0) {
      throw invalid_file_format_exception{fmt::format("Can't parse file '{}'!", filename)};
    }

    // update gamma
    if (gamma == 0) {
        gamma = 1. / num_features;
    }

    fmt::print("Read {} data points with {} features.\n", num_data_points, num_features);
}

//read ARF file
void CSVM::arffParser(const std::string_view filename) {
    std::ifstream file(filename.data());
    if (file.fail()) {
      throw file_not_found_exception{fmt::format("Couldn't find file: '{}'!", filename)};
    }

    std::string line, escape = "@";
    std::istringstream line_iss;
    std::vector<real_t> vline;
    std::string token;
    while (std::getline(file, line, '\n')) {
        if (line.compare(0, 1, "@") != 0 && line.size() > 1) {
            line_iss.str(line);
            while (std::getline(line_iss, token, ',')) {
                vline.push_back(util::convert_to<real_t, invalid_file_format_exception>(token));
            }
            line_iss.clear();
            if (vline.size() > 0) {
                value.push_back(vline.back());
                vline.pop_back();
                data.push_back(std::move(vline));
            }
            vline.clear();
        } else {
            std::cout << line;
        }
    }
    num_data_points = data.size();
    num_features = data[0].size();
    if (gamma == 0) {
        gamma = 1. / num_features;
    }
}

void CSVM::writeModel(const std::string_view model_name) { //TODO: idea: save number of Datapoint in input file ->  copy input file -> manipulate copy and dont rewrite whole File
    int nBSV = 0;
    int count_pos = 0;
    int count_neg = 0;
    for (int i = 0; i < alpha.size(); ++i) {
        if (value[i] > 0)
            ++count_pos;
        if (value[i] < 0)
            ++count_neg;
        if (alpha[i] == cost)
            ++nBSV;
    }
    //Terminal output
    if (info) {
        std::cout << "Optimization finished \n";
        std::cout << "nu = " << cost << "\n";
        std::cout << "obj = "
                  << "\t"
                  << ", rho " << -bias << "\n";
        std::cout << "nSV = " << count_pos + count_neg - nBSV << ", nBSV = " << nBSV << "\n";
        std::cout << "Total nSV = " << count_pos + count_neg << std::endl;
    }

    //Model file
    std::ofstream model(model_name.data(), std::ios::out | std::ios::trunc);
    model << "svm_type c_svc\n";
    switch (kernel) {
    case 0:
        model << "kernel_type "
              << "linear"
              << "\n";
        break;
    case 1:
        model << "kernel_type "
              << "polynomial"
              << "\n";
        break;
    case 2:
        model << "kernel_type "
              << "rbf"
              << "\n";
        break;
    default:
        throw std::runtime_error("Can not decide which kernel!");
    }
    model << "nr_class 2\n";
    model << "total_sv " << count_pos + count_neg << "\n";
    model << "rho " << -bias << "\n";
    model << "label "
          << "1"
          << " "
          << "-1"
          << "\n";
    model << "nr_sv " << count_pos << " " << count_neg << "\n";
    model << "SV\n";
    model << std::scientific;

    int count = 0;

#pragma omp parallel //num_threads(1) //TODO: fix bug. SV complete missing if more then 1 Thread
    {
        std::stringstream out_pos;
        std::stringstream out_neg;

// Alle SV Klasse 1
#pragma omp for nowait
        for (int i = 0; i < alpha.size(); ++i) {
            std::cout << value[i] << std::endl;
            if (value[i] > 0)
                out_pos << alpha[i] << " " << data[i] << "\n";
        }

#pragma omp critical
        {
            model << out_pos.str();
            count++;
#pragma omp flush(count, model)
        }

// Alle SV Klasse -1
#pragma omp for nowait
        for (int i = 0; i < alpha.size(); ++i) {
            if (value[i] < 0)
                out_neg << alpha[i] << " " << data[i] << "\n";
        }

        //Wait for all have written Class 1
        while (count < omp_get_thread_num()) {
        };

#pragma omp critical
        model << out_neg.str();
    }
    model.close();
}

} // namespace plssvm

// writeModel second version, TODO: implement correctly
//void CSVM::writeModel(std::string &model_name){
//  //  __itt_resume();
//  int nBSV = 0;
//  int count_pos = 0;
//  int count_neg = 0;
//  for(int i = 0; i < alpha.size(); ++i){
//    if(value[i] > 0) ++count_pos;
//    if(value[i] < 0) ++count_neg;
//    if(alpha[i] == cost) ++nBSV;
//
//  }
//  //Terminal Ausgabe
//  if(info){
//    std::cout << "Optimization finished \n";
//    std::cout << "nu = " << cost << "\n";
//    std::cout << "obj = " << "\t" << ", rho " << - bias << "\n";
//    std::cout << "nSV = " << count_pos + count_neg  - nBSV << ", nBSV = " << nBSV << "\n";
//    std::cout << "Total nSV = " << count_pos + count_neg << std::endl;
//  }
//  return;
//  //Model Datei
//  const unsigned int length = 1048576;
//  char buffer[length];
//  std::ofstream model;
//  model.rdbuf()->pubsetbuf(buffer, length);
//  model.open(model_name, std::ios::out | std::ios::trunc);
//  model << "svm_type c_svc\n";
//  switch(kernel){
//    case 0: model << "kernel_type " << "linear" << "\n";
//      break;
//    case 1: model << "kernel_type " << "polynomial" << "\n";
//      break;
//    case 2: model << "kernel_type " << "rbf" << "\n";
//      break;
//    default: throw std::runtime_error("Can not decide wich kernel!");
//  }
//  model << "nr_class 2\n";
//  model << "total_sv " << count_pos + count_neg << "\n";
//  model << "rho " << -bias << "\n";
//  model << "label " << "1" << " "<< "-1"<<"\n";
//  model << "nr_sv " << count_pos << " "<< count_neg<<"\n";
//  model << "SV\n";
//  // model << std::scientific;
//  model.unsetf(std::ios_base::floatfield);
//
//  int count = 0;
//  const size_t num_threads = 80; //omp_get_max_threads();
//  omp_set_num_threads(num_threads);
//  // auto start = std::chrono::high_resolution_clock::now();
//  // std::vector<char*> out_pos(80, new char[(data[0].size()*20 + 20)* data.size()/ omp_get_num_threads()]);
//  // auto stop = std::chrono::high_resolution_clock::now();
//  // std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(stop-start).count()<< std::endl;
//  // std::vector<char*> out_neg(80, new char[(data[0].size()*20 + 20)* data.size()/ omp_get_num_threads()]);
//
//#pragma omp parallel shared(count)
//  {
//    std::string out_pos;
//    out_pos.reserve((data[0].size()*20 + 20)* data.size()/ omp_get_num_threads());
//    std::string out_neg;
//    out_neg.reserve((data[0].size()*20 + 20)* data.size()/ omp_get_num_threads());
//    // char* ptr = out_pos[omp_get_thread_num()];
//    // unsigned long ptr_int = 0;
//    // std::stringstream out_neg;
//    // auto start = std::chrono::high_resolution_clock::now();
//    // Alle SV Klasse 1
//#pragma omp for nowait
//    for(int i = 0; i < alpha.size(); ++i){
//      if(value[i] > 0){
//        // sprintf(ptr,"%e %n", alpha[i], &ptr_int);
//        // ptr += ptr_int;
//        out_pos += std::to_string(alpha[i]) + ' ';
//        char buffer[20];
//        for(unsigned j = 0; j < data[i].size() ; ++j){
//          if(data[i][j] != 0.0 ){
//            // sprintf(ptr, "%i:%e %n",j,data[i][j], &ptr_int);
//            // ptr += ptr_int;
//            sprintf(buffer, "%i:%e ",j,data[i][j]);
//            // sprintf(buffer, "%i:%i.%ie^0 ",j,(int)data[i][j], static_cast<int>((data[i][j] - (int)data[i][j])*1000000)  );
//            out_pos += buffer;
//          } //out << i << ":" << vec[i] << " ";
//        }
//        out_pos += '\n';
//        //  *ptr = '\n';
//        //  ++ptr;
//        // out_pos << alpha[i]  << " " << buffer << '\n';
//        //  out_pos << alpha[i]  << " " << data[i] << "\n";
//
//      }
//    }
//    // auto stop = std::chrono::high_resolution_clock::now();
//    // std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(stop-start).count()<< std::endl;
//#pragma omp critical
//    {
//      // auto start = std::chrono::high_resolution_clock::now();
//      model << out_pos;
//#pragma omp flush (model)
//      // auto stop = std::chrono::high_resolution_clock::now();
//      // std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(stop-start).count()<< std::endl;
//    }
//
//#pragma omp atomic
//    count++;
//    // #pragma omp single
//    // {
//    // 	auto start = std::chrono::high_resolution_clock::now();
//    // 	for(auto i : out_pos)
//    // 		model << i;
//    // 	// count++;
//    // 	#pragma omp flush ( model)
//    // 	auto stop = std::chrono::high_resolution_clock::now();
//    // 	std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(stop-start).count()<< std::endl;
//    // }
//    // Alle SV Klasse -1
//    // #pragma omp for schedule(guided)
//    // for(int i = 0; i < alpha.size(); ++i){
//    // 	if(value[i] < 0) out_neg << alpha[i]  << " " << data[i] << "\n";
//    // }
//
//    {
//      // auto start = std::chrono::high_resolution_clock::now();
//      // Alle SV Klasse 1
//#pragma omp for nowait schedule(guided,4)
//      for(int i = 0; i < alpha.size(); ++i){
//        if(value[i] > 0){
//          out_pos += std::to_string(alpha[i]) + ' ';
//          char buffer[20];
//          for(unsigned j = 0; j < data[i].size() ; ++j){
//            if(data[i][j] != 0.0 ){
//              // sprintf(ptr, "%i:%e %n",j,data[i][j], &ptr_int);
//              // ptr += ptr_int;
//              sprintf(buffer, "%i:%e ",j,data[i][j]);
//              // sprintf(buffer, "%i:%i.%ie^0 ",j,(int)data[i][j], static_cast<int>((data[i][j] - (int)data[i][j])*1000000)  );
//              out_pos += buffer;
//            } //out << i << ":" << vec[i] << " ";
//          }
//          out_pos += '\n';
//
//        }
//      }
//      // auto stop = std::chrono::high_resolution_clock::now();
//      // std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(stop-start).count()<< std::endl;
//    }
//
//    //Wait for all have writen Klass 1
//    while(count < num_threads) {};
//
//
//#pragma omp critical
//    model << out_neg;
//  }
//  model.close();
//
//}