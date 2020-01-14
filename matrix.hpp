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
    virtual T& Entry(unsigned int i, unsigned int j) {};
    T& operator ()(unsigned int i, unsigned int j) { return Entry(i,j);}

    
    
    friend ostream& operator << (ostream& os, Matrix<T>& mat) {
        for (int i=0; i<mat.Size(); i++) {
            for (int j=0; j<mat.Size(); j++) {
                os<<mat(i,j)<<" ";
            }
            os<<"\n";
        }
        return os;
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

            return *(data[j]+i);
        }
        else {
            return *(data[i]+j);
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