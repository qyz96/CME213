#include <iostream>
#include <algorithm>
#include <vector>
#include <list>
#include <numeric>
#include <stdexcept>


/**********  Q4a: DAXPY **********/
template <typename T>
std::vector<T> daxpy(T a, const std::vector<T>& x, const std::vector<T>& y)
{
    // TODO
    return std::vector<T>();
}


/**********  Q4b: All students passed **********/
constexpr double HOMEWORK_WEIGHT = 0.20;
constexpr double MIDTERM_WEIGHT = 0.35;
constexpr double FINAL_EXAM_WEIGHT = 0.45;

struct Student
{
    double homework;
    double midterm;
    double final_exam;

    Student(double hw, double mt, double fe) : 
           homework(hw), midterm(mt), final_exam(fe) { }
};

bool all_students_passed(const std::vector<Student>& students, double pass_threshold) 
{
    // TODO 
    return false;
}


/**********  Q4c: Odd first, even last **********/
void sort_odd_even(std::vector<int>& data)
{
    // TODO
}

/**********  Q4d: Sparse matrix list sorting **********/
template <typename T>
struct SparseMatrixCoordinate
{
    int row;
    int col;
    T data;
    
    SparseMatrixCoordinate(int r, int c, T d) :
        row(r), col(c), data(d) { }
};

template <typename T>
void sparse_matrix_sort(std::list<SparseMatrixCoordinate<T>>& list) 
{
    // TODO
}

int main() 
{    
    // Q4a test
    const int Q4_A = 2;
    const std::vector<int> q4a_x = {-2, -1, 0, 1, 2};
    const std::vector<int> q4_y = {-2, -1, 0, 1, 2};

    // TODO: Verify your Q4a implementation

    // Q4b test
    std::vector<Student> all_pass_students = {
            Student(1., 1., 1.),
            Student(0.6, 0.6, 0.6),
            Student(0.8, 0.65, 0.7)};

    std::vector<Student> not_all_pass_students = {
            Student(1., 1., 1.),
            Student(0, 0, 0)};

    // TODO: Verify your Q4b implementation

    // Q4c test
    std::vector<int> odd_even_sorted = {-5, -3, -1, 1, 3, -4, -2, 0, 2, 4};

    // TODO: Verify your Q4c implementation

    // Q4d test
    std::list<SparseMatrixCoordinate<int>> sparse = {
            SparseMatrixCoordinate<int>(2, 5, 1),
            SparseMatrixCoordinate<int>(2, 2, 2),
            SparseMatrixCoordinate<int>(3, 4, 3)};

    // TODO: Verify your Q4d implementation

    return 0;
}
