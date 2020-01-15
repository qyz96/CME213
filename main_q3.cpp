#include <iostream>
#include <random>
#include <set>

// TODO: add your function here //


unsigned int CountData(const std::set<double>& data, double low, double high) {

    if (low > high) {std::cerr<<"Low value higher than high value!\n";}
    std::set<double>::iterator it1=data.lower_bound (low);                //       ^
    std::set<double>::iterator it2=data.upper_bound (high); 
    std::set<double>::iterator it;
    unsigned int count=0;
    for (it=it1; it != it2; it++) {
        count++;
    }
    return count;
}

int main()
{
    // Test with N(0,1) data.
    std::cout << "Generating N(0,1) data" << std::endl;
    std::set<double> data;
    double low=0.2;
    double high=0.8;
    unsigned int n=100;
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0, 1.0);
    for (unsigned int i = 0; i < n; ++i)
        data.insert(distribution(generator));
    for (std::set<double>::iterator it=data.begin(); it != data.end(); it++) {
        std::cout<<*it<<" ";
    }

    // TODO: print out number of points in [2, 10] //
    std::cout<<CountData(data,low, high)<<" values between "<<low<<" and "<<high<<".\n";
    return 0;
}
