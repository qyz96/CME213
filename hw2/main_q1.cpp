#include <iostream>
#include <random>
#include <vector>

#include <math.h>
#include <omp.h>
#include <stdlib.h>
#include <sys/time.h>

#include "tests_q1.h"

typedef unsigned int uint;
const uint kMaxInt = 100;
const uint kSize = 30000000;

std::vector<uint> serialSum(const std::vector<uint>& v) {
    std::vector<uint> sums(2);
    // TODO
    for (auto& it:v) {
        if (it%2==0) {
            sums[0]+=it;
        }
        else {
            sums[1]+=it;
        }
    }
    return sums;
}

std::vector<uint> parallelSum(const std::vector<uint>& v) {
    std::vector<uint> sums(2);
    uint sum1,sum2;
    #pragma omp parallel for default(shared) reduction(+:sum1, sum2)
    for (auto& it:v) {
        if (it%2==0) {
            sum1+=it;
            //printf("Thread %d works on element\n", omp_get_thread_num());
        }
        else {
            sum2+=it;
        }
    }
    
    sums[0]=sum1;
    sums[1]=sum2;
    return sums;
}

std::vector<uint> initializeRandomly(const uint size, const uint max_int) {
    std::vector<uint> res(size);
    std::default_random_engine generator;
    std::uniform_int_distribution<uint> distribution(0, max_int);

    for(uint i = 0; i < size; ++i) {
        res[i] = distribution(generator);
    }

    return res;
}

int main() {

    // You can uncomment the line below to make your own simple tests
    // std::vector<uint> v = ReadVectorFromFile("vec");
    std::vector<uint> v = initializeRandomly(kSize, kMaxInt);

    std::cout << "Parallel" << std::endl;
    struct timeval start, end;
    gettimeofday(&start, NULL);
    std::vector<uint> sums = parallelSum(v);
    std::cout << "Sum Even: " << sums[0] << std::endl;
    std::cout << "Sum Odd: " << sums[1] << std::endl;
    gettimeofday(&end, NULL);
    double delta = ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec -
                    start.tv_usec) / 1.e6;
    std::cout << "Time: " << delta << std::endl;

    std::cout << "Serial" << std::endl;
    gettimeofday(&start, NULL);
    std::vector<uint> sumsSer = serialSum(v);
    std::cout << "Sum Even: " << sumsSer[0] << std::endl;
    std::cout << "Sum Odd: " << sumsSer[1] << std::endl;
    gettimeofday(&end, NULL);
    delta = ((end.tv_sec  - start.tv_sec) * 1000000u + end.tv_usec -
             start.tv_usec) / 1.e6;
    std::cout << "Time: " << delta << std::endl;

    bool success = true;
    EXPECT_VECTOR_EQ(sums, sumsSer, &success);
    PRINT_SUCCESS(success);

    return 0;
}

