#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>

int predict_probability = 0;
bool info = true;

void exit_input_error(int line_num)
{
    if(info){
        std::cerr << "Wrong input format at line " << line_num << std::endl;
    }
	exit(1);
}

void exit_with_help()
{
    if(info){
        std::cout << "Usage: svm-predict [options] test_file model_file output_file\n";
        std::cout << "options:\n";
        std::cout << "not supportet jet: -b probability_estimates: whether to predict probability estimates, 0 or 1 (default 0); for one-class SVM only 0 is supported\n";
        std::cout << "-q : quiet mode (no outputs)" << std::endl;
    }
	exit(1);
}
void generate_w(std::vector<double> &w, std::ifstream &model){
    std::string line;
    int  totalSV, class1SV_nr, class2SV_nr, line_nr = 0;
    double rho;
    std::string temp;
    //svm_type TODO
    ++line_nr;
    if(!std::getline(model, line)) exit_input_error(line_nr);

    //kernel_type 
    ++line_nr;
    if(std::getline(model, line)){
       std::cout << "kernel type: "<<line.substr(line.find(" "))<< std::endl;
    } else{
        exit_input_error(line_nr);
    }
    
    //nr classes
    ++line_nr;
    if(std::getline(model, line)){
        if(std::stoi(line.substr(line.find(" ")),nullptr) != 2) {
            if(info)std::cerr << "Momentan nur für 2 Klassen implementiert" << std::endl;
            exit_input_error(line_nr);
        }
    } else{
        exit_input_error(line_nr);
    }

    //total SV
    ++line_nr;
    if(std::getline(model, line)){
        totalSV = std::stoi(line.substr(line.find(" ")),nullptr);
    } else{
        exit_input_error(line_nr);
    }

    //rho
    ++line_nr;
    if(std::getline(model, line)){
        rho = std::stod(line.substr(line.find(" ")),nullptr);
    } else{
        exit_input_error(line_nr);
    }

    //label
    ++line_nr;
    if(std::getline(model, line)){
        std::string token = line.substr(line.find(" ") + 1);
        std::cout << "label 1: "<<token.substr(0,token.find(" ")) << std::endl;
        std::cout << "label 2: "<<token.substr(token.find(" ")) << std::endl;
    } else{
        exit_input_error(line_nr);
    }

    //nr_sv
    ++line_nr;
    if(std::getline(model, line)){
        std::string token = line.substr(line.find(" ") + 1);
        class1SV_nr = std::stoi(token.substr(0,token.find(" ")) , nullptr);
        class2SV_nr = std::stoi(token.substr(token.find(" ")) , nullptr);
        if(class1SV_nr + class2SV_nr != totalSV){
            if(info) std::cerr << "Anzahl der SV stimmt nich überein" << std::endl;
            exit_input_error(line_nr);
        }
    } else{
        exit_input_error(line_nr);
    }

    //SV
    ++line_nr;
    if(!std::getline(model, line)) exit_input_error(line_nr);


    //TODO line_nr
    std::stringstream iss;
    std::string token;
    while (std::getline(model, line)){
        ++line_nr;
        iss.str(line);
        std::getline(iss, token, ' ');
        double fact = stod(token, nullptr);
        while (std::getline(iss, token, ' ')){
            int index = std::stoi(token.substr(0, token.find(":")),nullptr);
            double value = std::stod(token.substr(token.find(":")+1),nullptr);
            if(w.size() < index+1) w.resize(index + 1);
            w[index] =  fact * value;
        }
        iss.clear();
    }
}


void predict(std::ifstream &input, std::ifstream &model, std::ofstream &output){
    std::vector<double> w {};
    generate_w(w, model);
  

}

int main(int argc, char **argv){
    // parse options
    int i;
    for(i=1 ; i<argc; ++i)
    {
        if(argv[i][0] != '-') break;
        ++i;
        switch(argv[i-1][1])
        {
            case 'b':
                predict_probability = atoi(argv[i]);
                break;
            case 'q':
                info = false;
                --i;
                break;
            default:
                std::cerr << "Unknown option: -" << argv[i-1][1] << std::endl;
                exit_with_help();
        }
    }
    if(i>=argc-2) exit_with_help();

    std::ifstream input(argv[i]);
    if(!input){
        std::cerr << "can't open input file " << argv[i] << std::endl;
        exit(1);
    }

    std::ifstream model(argv[i+1]);
    if(!model){
        std::cerr << "can't open model file " << argv[i+1] << std::endl;
        exit(1);
    }

    std::ofstream output(argv[i+2],  std::ios::trunc);
    if(!output){
        std::cerr << "can't open output file " << argv[i+2] << std::endl;
        exit(1);
    }

    predict(input, model, output);
    input.close();
    model.close();
    output.close();
}



