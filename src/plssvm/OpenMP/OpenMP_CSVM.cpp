#include <chrono>
#include <omp.h>
#include <plssvm/OpenMP/OpenMP_CSVM.hpp>
#include <plssvm/operators.hpp>

namespace plssvm {

OpenMP_CSVM::OpenMP_CSVM(real_t cost_,
                         real_t epsilon_,
                         unsigned kernel_,
                         real_t degree_,
                         real_t gamma_,
                         real_t coef0_,
                         bool info_) : CSVM(cost_, epsilon_, kernel_, degree_, gamma_, coef0_, info_) {}

void OpenMP_CSVM::learn() {
    std::vector<real_t> q;
    std::vector<real_t> b = value;
#pragma omp parallel sections
    {
#pragma omp section // generate q
        {
            q.reserve(data.size());
            for (int i = 0; i < data.size() - 1; ++i) {
                q.emplace_back(kernel_function(data.back(), data[i]));
            }
        }
#pragma omp section // generate right side from eguation
        {
            b.pop_back();
            b -= value.back();
        }
#pragma omp section // generate botom right from A
        {
            QA_cost = kernel_function(data.back(), data.back()) + 1 / cost;
        }
    }

    std::cout << "start CG" << std::endl;
    //solve minimization
    alpha = CG(b, 1000, epsilon);

    alpha.emplace_back(-sum(alpha));
    bias = value.back() - QA_cost * alpha.back() - (q * alpha);
}

real_t CSVM::kernel_function(real_t *xi, real_t *xj, int dim) { //TODO: kernel as template
    switch (kernel) {
    case 0:
        return mult(xi, xj, dim);
    case 1:
        return std::pow(gamma * mult(xi, xj, dim) + coef0, degree);
    case 2: {
        real_t temp = 0;
        for (int i = 0; i < dim; ++i) {
            temp += (xi[i] - xj[i]);
        }
        return exp(-gamma * temp * temp);
    }
    default:
        throw std::runtime_error("Can not decide wich kernel!");
    }
}
void OpenMP_CSVM::learn(std::string &filename, std::string &output_filename) {
    auto begin_parse = std::chrono::high_resolution_clock::now();
    if (filename.size() > 5 && endsWith(filename, ".arff")) {
        arffParser(filename);
    } else {
        libsvmParser(filename);
    }
    auto end_parse = std::chrono::high_resolution_clock::now();
    if (info) {
        std::clog << data.size() << " Datenpunkte mit Dimension " << num_features << " in "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end_parse - begin_parse).count()
                  << " ms eingelesen" << std::endl
                  << std::endl;
    }

    learn();
    auto end_learn = std::chrono::high_resolution_clock::now();
    if (info)
        std::clog << std::endl
                  << data.size() << " Datenpunkte mit Dimension " << num_features << " in "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end_learn - end_parse).count() << " ms gelernt"
                  << std::endl;

    writeModel(output_filename);
    auto end_write = std::chrono::high_resolution_clock::now();
    if (info) {
        std::clog << std::endl
                  << data.size() << " Datenpunkte mit Dimension " << num_features << " in "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end_write - end_learn).count() << " geschrieben"
                  << std::endl;
    } else if (times) {
        std::clog << data.size() << ", " << num_features << ", " << 0 << ", "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end_learn - end_parse).count() << ", "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end_write - end_learn).count() << std::endl;
    }
}

std::vector<real_t> OpenMP_CSVM::CG(const std::vector<real_t> &b, const int imax, const real_t eps) {
    std::vector<real_t> x(b.size(), 1);
    real_t *datlast = &data.back()[0];
    static const size_t dim = data.back().size();
    static const size_t dept = b.size();

    //r = b - (A * x)
    ///r = b;
    real_t r[dept];
    std::copy(b.begin(), b.end(), r);

    int bloksize = 64;

#pragma omp parallel for collapse(2) schedule(dynamic, 8)
    for (int i = 1; i < (dept + bloksize); i = i + bloksize) {
        for (int j = 0; j < (dept + bloksize); j = j + bloksize) {

            for (int ii = 0; ii < bloksize && ii + i < dept; ++ii) {
                for (int jj = 0; jj < bloksize && jj + j < dept; ++jj) {
                    if (ii + i > jj + j) {
                        real_t temp = kernel_function(&data[ii + i][0], &data[jj + j][0], dim) - kernel_function(datlast, &data[ii + i][0], dim);
#pragma omp atomic
                        r[jj + j] -= temp;
#pragma omp atomic
                        r[ii + i] -= temp;
                    }
                }
            }
        }
    }

#pragma omp parallel for schedule(dynamic, 8)
    for (int i = 0; i < dept; ++i) {
        real_t kernel_dat_and_cost = kernel_function(datlast, &data[i][0], dim) - QA_cost;
#pragma omp atomic
        r[i] -= kernel_function(&data[i][0], &data[i][0], dim) - kernel_function(datlast, &data[i][0], dim) + 1 / cost - (b.size() - i) * kernel_dat_and_cost;
        for (int j = i + 1; j < dept; ++j) {
#pragma omp atomic
            r[j] = r[j] + kernel_dat_and_cost;
        }
    }

    std::cout << "r= b-Ax" << std::endl;
    real_t d[b.size()] = {0};

    std::memcpy(d, r, dept * sizeof(real_t));

    real_t delta = mult(r, r, sizeof(r) / sizeof(real_t));
    const real_t delta0 = delta;
    real_t alpha, beta;

    for (int run = 0; run < imax; ++run) {
        std::cout << "Start Iteration: " << run << std::endl;
        //Ad = A * d
        real_t Ad[dept] = {0.0};

#pragma omp parallel for collapse(2) schedule(dynamic, 8)
        for (int i = 0; i < b.size(); i += bloksize) {
            for (int j = 0; j < b.size(); j += bloksize) {

                real_t temp_data_i[bloksize][data[0].size()];
                real_t temp_data_j[bloksize][data[0].size()];
                for (int ii = 0; ii < bloksize; ++ii) {

                    if (ii + i < b.size())
                        std::copy(data[ii + i].begin(), data[ii + i].end(), temp_data_i[ii]);
                    if (ii + j < b.size())
                        std::copy(data[ii + j].begin(), data[ii + j].end(), temp_data_j[ii]);
                }
                for (int ii = 0; ii < bloksize && ii + i < b.size(); ++ii) {
                    for (int jj = 0; jj < bloksize && jj + j < b.size(); ++jj) {

                        if (ii + i > jj + j) {
                            real_t temp = kernel_function(temp_data_i[ii], temp_data_j[jj], dim) - kernel_function(datlast, temp_data_j[jj], dim);
#pragma omp atomic
                            Ad[jj + j] += temp * d[ii + i];
#pragma omp atomic
                            Ad[ii + i] += temp * d[jj + j];
                        }
                    }
                }
            }
        }

#pragma omp parallel for schedule(dynamic, 8)
        for (int i = 0; i < b.size(); ++i) {
            real_t kernel_dat_and_cost = kernel_function(datlast, &data[i][0], dim) - QA_cost;
#pragma omp atomic
            Ad[i] += (kernel_function(&data[i][0], &data[i][0], dim) - kernel_function(datlast, &data[i][0], dim) + 1 / cost - kernel_dat_and_cost) * d[i];
            for (int j = 0; j < i; ++j) {
#pragma omp atomic
                Ad[j] -= kernel_dat_and_cost * d[i];
#pragma omp atomic
                Ad[i] -= kernel_dat_and_cost * d[j];
            }
        }

        alpha = delta / mult(d, Ad, sizeof(d) / sizeof(real_t));
        x += mult(alpha, d, sizeof(d) / sizeof(real_t));
        //r = b - (A * x)
        ///r = b;
        std::copy(b.begin(), b.end(), r);

#pragma omp parallel for collapse(2) schedule(dynamic, 8)
        for (int i = 0; i < (b.size() + bloksize); i += bloksize) {
            for (int j = 0; j < (b.size() + bloksize); j += bloksize) {

                for (int ii = 0; ii < bloksize && ii + i < b.size(); ++ii) {
                    for (int jj = 0; jj < bloksize && jj + j < b.size(); ++jj) {
                        if (ii + i > jj + j) {
                            real_t temp = kernel_function(&data[ii + i][0], &data[jj + j][0], dim) - kernel_function(datlast, &data[jj + j][0], dim);
#pragma omp atomic
                            r[jj + j] -= temp * x[ii + i];
#pragma omp atomic
                            r[ii + i] -= temp * x[jj + j];
                        }
                    }
                }
            }
        }

#pragma omp parallel for schedule(dynamic, 8)
        for (int i = 0; i < b.size(); ++i) {
            real_t kernel_dat_and_cost = kernel_function(datlast, &data[i][0], dim) - QA_cost;
#pragma omp atomic
            r[i] -= (kernel_function(&data[i][0], &data[i][0], dim) - kernel_function(datlast, &data[i][0], dim) + 1 / cost - kernel_dat_and_cost) * x[i];
            for (int j = 0; j < i; ++j) {
#pragma omp atomic
                r[j] += kernel_dat_and_cost * x[i];
#pragma omp atomic
                r[i] += kernel_dat_and_cost * x[j];
            }
        }

        delta = mult(r, r, b.size());
        //break;
        if (delta < eps * eps * delta0)
            break;
        beta = -mult(r, Ad, b.size()) / mult(d, Ad, b.size());
        add(mult(beta, d, sizeof(d) / sizeof(real_t)), r, d, sizeof(d) / sizeof(real_t));
    }

    return x;
}

} // namespace plssvm