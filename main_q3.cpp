#include <iostream>
#include <random>
#include <set>

// TODO: add your function here //


int main()
{
    // Test with N(0,1) data.
    std::cout << "Generating N(0,1) data" << std::endl;
    std::set<double> data;
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0, 1.0);
    for (unsigned int i = 0; i < 1000; ++i)
        data.insert(distribution(generator));

    // TODO: print out number of points in [2, 10] //

    return 0;
}
