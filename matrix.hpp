#ifndef _MATRIX_HPP
#define _MATRIX_HPP
#include<iostream>
#include<algorithm>
#include <stdexcept>
using namespace std;

template <typename T>
class Matrix
{
    public :
    Matrix(unsigned int n): size(n) {}
    T& Entry(unsigned int i, unsigned int j) {};
    const T& Entry(unsigned int i, unsigned int j) const{};
    const T& operator ()(unsigned int i, unsigned int j) const { return Entry(i,j);}
    T& operator ()(unsigned int i, unsigned int j) { return Entry(i,j);}

    
    
    friend ostream& operator << (ostream& os, Matrix<T>& mat) {
        for (int i=0; i<mat.size; i++) {
            for (int j=0; j<mat.size; j++) {
                os<<mat(i,j)<<" ";
            }
            os<<"\n";
        }
        return os;
    }
        unsigned int l0norm() {
        unsigned int norm=0;
        for (unsigned int i=0; i<size; i++) {
            for (unsigned int j=0; j<size; j++) {
                if ((*this)(i,j)>0) {
                    norm++;
                }
            }
        }
        return norm;
    }


    private:
    unsigned int size;
};
template <typename T>
class MatrixSymmetric: public Matrix<T>
{
    public:
    MatrixSymmetric(int n): Matrix<T>(n), data(new T*[n]) {
        for (int i=0; i<n; i++) {
            data[i]=new T[n-i];
        }
    }
    T& Entry(unsigned int i, unsigned int j) {

        if (i>=j) {

            return *(data[j]+i-j);
        }
        else {
            return *(data[i]+j-i);
        }
    }

    

    ~MatrixSymmetric() {

        
        for (int i=0; i<1; i++) {
            delete[] data[i];
            //data[i]=nullptr;
        }
        delete [] data;
        
    }

    private:
    T** data;

};

#endif /* MATRIX_HPP */