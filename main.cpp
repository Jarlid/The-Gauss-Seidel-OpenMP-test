#include <cfloat>
#include <cmath>
#include <random>

#include <vector>

#include <iostream>
#include <fstream>

#include <omp.h>
#include <chrono>

double example_f(double x, double y) {
    return 0;
}

double example_g(double x, double y) {
    if (x == 0)
        return 100 - 200 * y;
    if (x == 1)
        return 200 * y - 100;
    if (y == 0)
        return 100 - 200 * x;
    if (y == 1)
        return 200 * x - 100;
    exit(823476);
}


void to_picture(int N, const std::vector<std::vector<double>>& data, const std::string& filename) {
    std::vector<uint8_t> printable_data(3 * (N + 2) * (N + 2), 0);

    double max_data_abs = 0;
    for (int i = 0; i < N + 2; ++i)
        for (int j = 0; j < N + 2; ++j)
            max_data_abs = fmax(max_data_abs, fabs(data[i][j]));

    for (int i=0; i<N+2; i++ ) {
        for (int j=0; j<N+2; j++ ) {
            int it = 3 * (j + i * (N + 2));
            if (data[i][j] < 0)
                printable_data[it + 2] = (uint8_t) fabs(data[i][j] * 255 / max_data_abs);
            else
                printable_data[it + 0] = (uint8_t) fabs(data[i][j] * 255 / max_data_abs);
        }
    }

    std::ofstream out_stream(filename, std::ios_base::binary);
    out_stream << "P6" << std::endl;
    out_stream << N + 2 << " " << N + 2 << std::endl;
    out_stream << 255 << std::endl;
    out_stream.write((char*) printable_data.data(), 3 * (N + 2) * (N + 2));
}


std::vector<std::vector<double>> initialize_data(int N, double (*g)(double, double)) {
    std::vector<std::vector<double>> data(N + 2, std::vector<double>(N + 2, 0));
    double min_data = DBL_MAX, max_data = -DBL_MAX;

    for (int i = 0; i < N + 2; ++i) {
        double x = (double) i / (N + 1);

        data[i][0] = g(x, 0);
        data[0][i] = g(0, x);
        data[i][N + 1] = g(x, 1);
        data[N + 1][i] = g(1, x);

        std::vector<double> new_data{data[i][0], data[0][i], data[i][N + 1], data[N + 1][i]};
        for (double new_data_sample : new_data) {
            min_data = fmin(min_data, new_data_sample);
            max_data = fmax(max_data, new_data_sample);
        }
    }

    std::default_random_engine engine{};
    std::uniform_real_distribution<double> distribution =
            std::uniform_real_distribution<double>(min_data, max_data);

    for (int i = 1; i < N + 1; ++i)
        for (int j = 1; j < N + 1; ++j) {
            data[i][j] = distribution(engine);
        }

    return data;
}


void basic_calculation(int N, double eps, double (*f)(double, double), double (*g)(double, double)) {
    std::vector<std::vector<double>> data = initialize_data(N, g);

    int loop_cnt = 0;
    double max_delta;

    auto begin = std::chrono::steady_clock::now();

    do {
        ++loop_cnt;
        max_delta = 0;

        for (int i = 1; i < N + 1; ++i) {
            for (int j = 1; j < N + 1; ++j) {
                double data_i_j = data[i][j];
                data[i][j] = 0.25 * (data[i - 1][j] + data[i + 1][j] + data[i][j - 1] + data[i][j + 1] -
                        f((double) i / (N + 1), (double) j / (N + 1)) / (N + 1) / (N + 1));

                double delta = fabs(data_i_j - data[i][j]);
                max_delta = fmax(max_delta, delta);
            }
        }
    } while (max_delta > eps);

    auto end = std::chrono::steady_clock::now();

    std::cout << "Loop count: " << loop_cnt << std::endl;
    std::cout << "Time:       "
              << (double) std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1'000'000
              << " s" << std::endl;

    to_picture(N, data, "basic_picture.ppm");
}


void parallel_calculation(int N, int NB, int thread_num, double eps,
                          double (*f)(double, double), double (*g)(double, double)) {
    int block_num = (N + NB - 1) / NB;
    std::vector<std::vector<double>> data = initialize_data(N, g);

    omp_lock_t max_delta_lock;
    omp_init_lock(&max_delta_lock);

    int loop_cnt = 0;
    double max_delta;

    omp_set_num_threads(thread_num);

    auto begin = std::chrono::steady_clock::now();

    do {
        ++loop_cnt;
        max_delta = 0;
        std::vector<double> local_max_delta(N + 2, 0);

        for (int diagonal = 0; diagonal < block_num; ++diagonal) {
            #pragma omp parallel for default(none) shared(N, NB, diagonal, data, f, local_max_delta)
            for (int block_i = 0; block_i < diagonal + 1; ++block_i) {
                int block_j = diagonal - block_i;

                for (int inner_i = 0; inner_i < NB; ++inner_i) {
                    int i = 1 + block_i * NB + inner_i;
                    if (i > N)
                        continue;

                    for (int inner_j = 0; inner_j < NB; ++inner_j) {
                        int j = 1 + block_j * NB + inner_j;
                        if (j > N)
                            continue;

                        double data_i_j = data[i][j];
                        data[i][j] = 0.25 * (data[i - 1][j] + data[i + 1][j] + data[i][j - 1] + data[i][j + 1] -
                                             f((double) i / (N + 1), (double) j / (N + 1)) / (N + 1) / (N + 1));

                        double delta = fabs(data_i_j - data[i][j]);
                        local_max_delta[i] = fmax(local_max_delta[i], delta);
                    }
                }
            }
        }

        for (int diagonal = block_num - 2; diagonal >= 0; --diagonal) {
            #pragma omp parallel for default(none) shared(N, NB, block_num, diagonal, data, f, local_max_delta)
            for (int block_i = block_num - diagonal - 1; block_i < block_num; ++block_i) {
                int block_j = 2 * block_num - 2 - diagonal - block_i;

                for (int inner_i = 0; inner_i < NB; ++inner_i) {
                    int i = 1 + block_i * NB + inner_i;
                    if (i > N)
                        continue;

                    for (int inner_j = 0; inner_j < NB; ++inner_j) {
                        int j = 1 + block_j * NB + inner_j;
                        if (j > N)
                            continue;

                        double data_i_j = data[i][j];
                        data[i][j] = 0.25 * (data[i - 1][j] + data[i + 1][j] + data[i][j - 1] + data[i][j + 1] -
                                             f((double) i / (N + 1), (double) j / (N + 1)) / (N + 1) / (N + 1));

                        double delta = fabs(data_i_j - data[i][j]);
                        local_max_delta[i] = fmax(local_max_delta[i], delta);
                    }
                }
            }
        }

        #pragma omp parallel for default(none) shared(N, NB, max_delta, local_max_delta, max_delta_lock)
        for (int big_i = 1; big_i < N + 1; big_i += NB) {
            double current_max_delta = 0;
            for (int i = big_i; i < big_i + NB and i < N + 1; ++i)
                current_max_delta = fmax(current_max_delta, local_max_delta[i]);
            omp_set_lock(&max_delta_lock);
            max_delta = fmax(max_delta, current_max_delta);
            omp_unset_lock(&max_delta_lock);
        }

    } while (max_delta > eps);

    auto end = std::chrono::steady_clock::now();

    std::cout << "Loop count: " << loop_cnt << std::endl;
    std::cout << "Time:       "
              << (double) std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1'000'000
              << " s" << std::endl;

    to_picture(N, data, "parallel_picture.ppm");
}


int main() {
    int N = 1000, NB = 100, thread_num = 4;
    double eps = 0.1;

    basic_calculation(N, eps, example_f, example_g);
    parallel_calculation(N, NB, thread_num, eps, example_f, example_g);

    return 0;
}
