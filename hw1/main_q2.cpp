#include <iostream>
#include <string>
#include <vector>

/* TODO: Make Matrix a pure abstract class with the 
 * public method:
 *     std::string repr()
 */
class Matrix 
{
    public:
    Matrix() {}
    virtual std::string repr()  { return "abstract matrix";}


};

/* TODO: Modify the following function so that it 
 * inherits from the Matrix class */
class SparseMatrix : public Matrix
{
public:
    SparseMatrix() {}
    std::string repr() 
    {
        return "sparse";
    }
};

/* TODO: Modify the following function so that it 
 * inherits from the Matrix class */
class ToeplitzMatrix : public Matrix
{
public:
    ToeplitzMatrix() {}
    std::string repr() 
    {
        return "toeplitz";
    }
};

/* TODO: This function should accept a vector of Matrices and call the repr 
 * function on each matrix, printing the result to standard out. 
 */
void PrintRepr(const std::vector<Matrix*>& data)
{
    for (unsigned int i=0; i<data.size(); i++) {
        std::cout<<data[i]->repr()<<"\n";
    }
}

/* TODO: Your main method should fill a vector with an instance of SparseMatrix
 * and an instance of ToeplitzMatrix and pass the resulting vector
 * to the PrintRepr function.
 */ 
int main() 
{
    int n=10;
    std::vector<Matrix*> data(n);
    for (unsigned int i=0; i<n-1; i++) {
        data[i]=new SparseMatrix;
    }
    data[9]=new ToeplitzMatrix;

    PrintRepr(data);

}
