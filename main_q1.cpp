#include "matrix.hpp"


int main()
{
    
    MatrixSymmetric<double> mat1(10);
    MatrixSymmetric<double> mat2(10);  
    
    for (unsigned int i=0; i<10; i++){
        for (unsigned int j=0; j<10; j++){
            mat1(i,j)=i+j;
            mat2(i,j)=2*i+3*j;
        }
    }
    
    //Matrix<double>* mat3;
    //*mat3=mat1+mat2;
    std::cout<<mat1;
    
    std::cout<<"L0 norm is "<<mat1.l0norm()<<"\n";
    return 0;
}