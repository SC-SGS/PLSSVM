#include "CSVM.hpp"

//Einlesen libsvm Dateien
void CSVM::libsvmParser(std::string &filename){
	unsigned maxsize = 0;
	std::ifstream file(filename);
	std::string line;
	maxsize = 0;
	std::stringstream iss;
	std::stringstream tokens;
	unsigned index;
	unsigned i;
	std::vector<double> vline;
	std::string token;
	while (std::getline(file, line))
	{
		iss.str(line);
		i = 0; 
		std::getline(iss, token, ' ');
		value.emplace_back(std::stod(token,nullptr) > 0 ? 1 : -1);
		while (std::getline(iss, token, ' ')){
			if(token != ""){
				tokens.str(token);
				std::getline(tokens, token, ':');
				index = std::stoul(token,nullptr);
				while (index > i++) vline.emplace_back(0.0);
				std::getline(tokens, token);	
				vline.emplace_back(std::stod(token, nullptr));
			}
			tokens.clear();
		}
		iss.clear();
		data.push_back(vline);
		vline.clear();
		maxsize = i > maxsize ? i : maxsize;
	}
	#pragma omp parallel for
	for (int i = 0; i < data.size(); ++i) {
		data[i].resize(maxsize, 0.0);
	}
	Nfeatures_data = maxsize;
	Ndatas_data = data.size();
	if(gamma == 0) gamma = 1/ Ndatas_data;
}


//Einlesen ARF Dateien
void CSVM::arffParser(std::string &filename){
	std::ifstream file(filename);
	std::string line, escape = "@";
	std::stringstream iss;
	std::vector<double> vline;
	std::string token;
	while (std::getline(file, line, '\n')) {
		if(line.compare(0, 1, "@") != 0 && line.size() > 1){
			iss.str(line);
			while(std::getline(iss, token, ',')){
				vline.push_back(std::stod(token, nullptr));
			}
			iss.clear();
			if(vline.size() > 0){
				value.push_back(vline.back());
				vline.pop_back();
				data.push_back(vline);
			}
			vline.clear();
		}else{
			std::cout << line;
		}
	
	}
	Ndatas_data = data.size();
	Nfeatures_data = data[0].size();

}

void CSVM::writeModel(std::string &model_name){
	int nBSV = 0;
	int count_pos = 0;
	int count_neg = 0;
	for(int i = 0; i < alpha.size(); ++i){
		if(value[i] > 0) ++count_pos;
		if(value[i] < 0) ++count_neg;
		if(alpha[i] == cost) ++nBSV;

	}
	//Terminal Ausgabe
	if(info){
		std::cout << "Optimization finished \n";
		std::cout << "nu = " << cost << "\n";
		std::cout << "obj = " << "\t" << ", rho " << - bias << "\n";
		std::cout << "nSV = " << count_pos + count_neg  - nBSV << ", nBSV = " << nBSV << "\n";
		std::cout << "Total nSV = " << count_pos + count_neg << std::endl;
	}

	//Model Datei
	std::ofstream model(model_name, std::ios::out | std::ios::trunc);
	model << "svm_type c_svc\n";
	switch(kernel){
		case 0: model << "kernel_type " << "linear" << "\n";
			break;
		case 1: model << "kernel_type " << "polynomial" << "\n";
			break;
		case 2: model << "kernel_type " << "rbf" << "\n";
			break;
		default: throw std::runtime_error("Can not decide wich kernel!");
	}
	model << "nr_class 2\n";
	model << "total_sv " << count_pos + count_neg << "\n";
	model << "rho " << -bias << "\n";
	model << "label " << "1" << " "<< "-1"<<"\n";
	model << "nr_sv " << count_pos << " "<< count_neg<<"\n";
	model << "SV\n";
	model << std::scientific;

	// Alle SV Klasse 1
	for(int i = 0; i < alpha.size(); ++i){
		if(value[i] > 0) model << alpha[i]  << " " << data[i] << "\n";
	}
	// Alle SV Klasse -1
	for(int i = 0; i < alpha.size(); ++i){
		if(value[i] < 0) model << alpha[i]  << " " << data[i] << "\n";
	}
	model.close();
	
}