#include "matrix.hpp"
#include <cassert>
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
    for (int j=0; j<data.Size(); j++) {
        for (int i=j; i<data.Size(); i++) {
            data(i,j)=f(i,j);
        }
    }
    return;
}

template <typename T>
unsigned int TestL0Norm(const Matrix<T>& data) {
    unsigned int l0=0;
    for (unsigned int j=0; j<data.Size(); j++) {
        for (unsigned int i=0; i<data.Size(); i++) {
            if (data(i,j) != 0) {
                l0++;
            }
        }
    }
    return l0;
}

template <typename T>
unsigned int TestAdd(const Matrix<T>& z, const Matrix<T>& x,const Matrix<T>& y) {
    for (unsigned int j=0; j<x.Size(); j++) {
        for (unsigned int i=0; i<x.Size(); i++) {
            if (z(i,j) != x(i,j)+y(i,j)) {
                return false;
            }
        }
    }
    return true;
}

template <typename T>
unsigned int TestSubtract(const Matrix<T>& z, const Matrix<T>& x,const Matrix<T>& y) {
    for (unsigned int j=0; j<x.Size(); j++) {
        for (unsigned int i=0; i<x.Size(); i++) {
            if (z(i,j) != x(i,j)-y(i,j)) {
                return false;
            }
        }
    }
    return true;
}

template <typename T>
unsigned int TestMultiply(const Matrix<T>& z, const Matrix<T>& x,const Matrix<T>& y) {
    for (unsigned int j=0; j<x.Size(); j++) {
        for (unsigned int i=0; i<x.Size(); i++) {
            T sum=0;
            for (unsigned int k=0; k<x.Size(); k++ ) {
                sum+=x(i,k)*y(k,j);
            }
            if (z(i,j) != sum) {
                return false;
            }
        }
    }
    return true;
}



/*This test code creates matrices with different values and sizes, 
    checks symmetry and arithmetic operations*/
int main()
{
    unsigned int n=5;
    MatrixSymmetric<double> mat_small(n);
    MatrixSymmetric<double> mat_small2(n);
    MatrixSymmetric<double> mat_large(2*n);  
    auto f1=[](int i, int j) {return (i+j+1);};
    auto f2=[](int i, int j){return (i-j-3);};
    AssignVal(mat_small, f1);
    AssignVal(mat_small2, f2);
    AssignVal(mat_large, f2);
    if (VerifySymmetry(mat_small)) {
        std::cout<<"Small matrix symmetry verified!\n";
    }
    if (VerifySymmetry(mat_large)) {
        std::cout<<"Large matrix symmetry verified!\n";
    }
    std::vector<Matrix<double>*> data(2);
    data[0]=&mat_small;
    data[1]=&mat_large;
    std::cout<<"Printing small matrix 1:\n"<<*data[0];
    std::cout<<"Printing small matrix 2:\n"<<mat_small2;
    std::cout<<"Printing large matrix:\n"<<*data[1];
    assert(data[1]->l0norm()==TestL0Norm(*data[1]));
    std::cout<<"L0 norm of large matrix is "<<data[1]->l0norm()<<"\n";
    assert(TestAdd(*data[0]+mat_small2, *data[0],mat_small2));
    std::cout<<"Adding mat_small and matsmall2:\n"<<*data[0]+mat_small2;
    assert(TestSubtract(*data[0]-mat_small2, *data[0],mat_small2));
    std::cout<<"Subtracting mat_small2 from mat_small:\n"<<*data[0]-mat_small2;
    assert(TestMultiply(*data[0]*mat_small2, *data[0],mat_small2));
    std::cout<<"Computing mat_small*mat_small2:\n"<<*data[0]*mat_small2;
    return 0;
}