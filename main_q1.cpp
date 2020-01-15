#include "matrix.hpp"

template <typename T>
bool VerifySymmetry(const MatrixSymmetric<T>& data) {
    for (unsigned int j=0; j<data.Size(); j++) {
        for (unsigned int i=j; i<data.Size(); i++) {
            if (data(i,j) != data(j,i)) {
                return false;
            }
        }
    }
    return true;
}

template <typename T, typename F>
void AssignVal(MatrixSymmetric<T>& data, F f) {
    for (unsigned int j=0; j<data.Size(); j++) {
        for (unsigned int i=j; i<data.Size(); i++) {
            data(i,j)=f(i,j);
        }
    }
    return;
}



int main()
{
    int n=2;
    MatrixSymmetric<double> mat_small(2);
    MatrixSymmetric<double> mat_large(20);  
    auto f1=[](unsigned int i, unsigned int j)->double {return (double)(1/(i+j+1));};
    auto f2=[](unsigned int i, unsigned int j)->double {return (double)(1/(i*i+j*j+1));};
    AssignVal(mat_small, f1);
    AssignVal(mat_large, f2);
    if (VerifySymmetry(mat_small)) {
        std::cout<<"Small matrix symmetry verified!\n";
    }
    if (VerifySymmetry(mat_large)) {
        std::cout<<"Large matrix symmetry verified!\n";
    }
    std::cout<<"Printing small matrix:\n"<<mat_small;
    std::cout<<"Printing large matrix:\n"<<mat_large;
    //std::vector<Matrix<double>*> data(2);
    std::cout<<"L0 norm of large matrix is "<<mat_large.l0norm()<<"\n";
    return 0;
}