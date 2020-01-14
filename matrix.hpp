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
    Matrix(): size(0), IsDense(false) {}
    //Initialize with dense matrix class
    Matrix(unsigned int n): size(n), IsDense(true), DensePtr(new T*[n]) {
        for (int i=0; i<n; i++) {
            DensePtr[i]=new T[n];
        }
    }
    //Initialize with abstract matrix class
    Matrix(unsigned int n, bool dense): size(n), IsDense(dense) {}


    ~Matrix() {
        if (IsDense) {
            for (int i=0; i<Size(); i++) {
            delete[] DensePtr[i];
        }
        delete[] DensePtr;
        }
    }
    virtual T& Entry(unsigned int i, unsigned int j) {
        if (IsDense) {
            return *(DensePtr[j]+i);
        }
    };
    virtual const T& Entry(unsigned int i, unsigned int j) const{
        if (IsDense) {
            return *(DensePtr[j]+i);
        }
    };
    const T& operator ()(unsigned int i, unsigned int j) const { return Entry(i,j);}
    T& operator ()(unsigned int i, unsigned int j) { return Entry(i,j);}

    
    
    friend ostream& operator << (ostream& os, const Matrix<T>& mat) {
        for (int i=0; i<mat.size; i++) {
            for (int j=0; j<mat.size; j++) {
                os<<mat(i,j)<<" ";
            }
            os<<"\n";
        }
        return os;
    }

    
    Matrix<T> operator + (const Matrix<T>& mat1) const {
        Matrix<T> output(mat1.Size());
        if (mat1.Size()!=this->Size()) {
            cerr << "Matrices should have the same sizes!\n";
            return output;
        }
        for (unsigned int i=0; i<mat1.Size(); i++) {
            for (unsigned int j=0; j<mat1.Size(); j++) {
                output(i,j)=mat1(i,j)+(*this)(i,j);
            }
        }
        
        return output;
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

    const unsigned int Size() const {
        return this->size;
    }


    private:
    T** DensePtr;
    bool IsDense;
    unsigned int size;
};
template <typename T>
class MatrixSymmetric: public Matrix<T>
{
    public:
    MatrixSymmetric(int n): Matrix<T>(n, false), data(new T*[n]) {
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

    const T& Entry(unsigned int i, unsigned int j) const {

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
        }
        delete [] data;
        
    }
    private:
    T** data;

};

#endif /* MATRIX_HPP */