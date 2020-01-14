#include "matrix.hpp"


int main()
{
    
    MatrixSymmetric<double> mat1(10);
    MatrixSymmetric<double> mat2(10);  
    int n=2;
    
    for (unsigned int i=0; i<n; i++){
        for (unsigned int j=0; j<n; j++){
            mat1(i,j)=i+j;
        }
    }
    
    //Matrix<double>* mat3;
    //*mat3=mat1+mat2;
    for (unsigned int i=0; i<n; i++){
        for (unsigned int j=0; j<n; j++){
            cout<<mat1(i,j)<<" ";
        }
        cout<<"\n";
    }
    cout<<mat1;
    //std::cout<<"L0 norm is "<<mat1.l0norm()<<"\n";
    return 0;
}