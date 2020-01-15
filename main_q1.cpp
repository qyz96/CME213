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

int main()
{
    int n=2;
    MatrixSymmetric<double> mat_small(2);
    MatrixSymmetric<double> mat_large(20);  
    
    for (unsigned int i=0; i<n; i++){
        for (unsigned int j=0; j<n; j++){
            mat1(i,j)=i+j;
            mat2(i,j)=3*3*i;
        }
    }
    std::vector<Matrix<double>*> data(2);
    data[0]=&mat1;
    data[1]=&mat2;
    cout<<mat1<<mat2;
    Matrix<double> mat3=*data[0]+*data[1];
    Matrix<double> mat4=*data[0]-*data[1];
    Matrix<double> mat5=(*data[0])*(*data[1]);
    cout<<mat3;
    cout<<mat4;
    cout<<mat5;
    std::cout<<"L0 norm is "<<mat1.l0norm()<<"\n";
    return 0;
}