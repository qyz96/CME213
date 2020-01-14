#include "matrix.hpp"


int main()
{
    int n=2;
    MatrixSymmetric<double> mat1(n);
    MatrixSymmetric<double> mat2(n);  
    
    for (unsigned int i=0; i<n; i++){
        for (unsigned int j=0; j<n; j++){
            mat1(i,j)=i+j;
            mat2(i,j)=3*3*i;
        }
    }
    
    Matrix<double> mat3=mat1+mat2;
    cout<<mat3;
    std::cout<<"L0 norm is "<<mat1.l0norm()<<"\n";
    return 0;
}