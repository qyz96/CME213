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
    if (x.size()!= y.size()) {
        std::cout<<"Adding vectors with different sizes!\n";
        exit(0);
    }
    unsigned int i=0;
    std::vector<T> z(x.size());
    std::for_each(z.begin(), z.end(), [&](T& s){s=a*x[i]+y[i];i++;});
    return z;
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
    return std::all_of(students.begin(), students.end(), [=](const Student& s){
        return (s.homework*HOMEWORK_WEIGHT+s.midterm*MIDTERM_WEIGHT+s.final_exam*FINAL_EXAM_WEIGHT>=pass_threshold);
    });
}


/**********  Q4c: Odd first, even last **********/
void sort_odd_even(std::vector<int>& data)
{
    // TODO
    std::sort(data.begin(),data.end(), [](int a, int b){
        if (a % 2 != 0 && b % 2 == 0) {
            return true;
        }
        else if (a % 2 ==0 && b % 2 != 0) {
            return false;
        }
        else {
            return a < b;
        }
    });
    return;
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
    list.sort([](SparseMatrixCoordinate<T> a, SparseMatrixCoordinate<T> b) {
        if (a.row < b.row) {
            return true;
        }
        else if (a.row == b.row && a.col<b.col) {
            return true;
        }
        else {
            return false;
        }
    });
}

int main() 
{    
    // Q4a test
    const int Q4_A = 2;
    const std::vector<int> q4a_x = {-2, -1, 0, 1, 2};
    const std::vector<int> q4_y = {-2, -1, 0, 1, 2};
    std::vector<int> z=daxpy(Q4_A, q4a_x, q4_y);
    std::cout<<"ax+y is:\n";
    bool q1=true;
    for (unsigned int i=0; i<5; i++) {
        if (z[i] != Q4_A*q4a_x[i]+q4_y[i]) {
            q1=false;
        }
    }
    if (q1) {
        std::cout<<"Q1 computation successful!\n";
    }
    else{
        std::cout<<"Q1 computation failed!\n";
    }
    // TODO: Verify your Q4a implementation

    // Q4b test
    std::vector<Student> all_pass_students = {
            Student(1., 1., 1.),
            Student(0.6, 0.6, 0.6),
            Student(0.8, 0.65, 0.7)};

    std::vector<Student> not_all_pass_students = {
            Student(1., 1., 1.),
            Student(0, 0, 0)};

    if (all_students_passed(all_pass_students, 0.6)) {
        std::cout<<"all_pass_students have passed the class!\n";
    }
    if (!all_students_passed(not_all_pass_students, 0.6)) {
        std::cout<<"not_all_pass_students have not passed the class!\n";
    }
    // TODO: Verify your Q4b implementation

    // Q4c test
    std::vector<int> odd_even_sorted = {-5, -4, -3, 2, -1, 0, 2, 5, 4, 1};

    sort_odd_even(odd_even_sorted);
    std::cout<<"Sorted odd even list\n";
    bool q3=true;
    auto f1=[](int a, int b){
        if (a % 2 != 0 && b % 2 == 0) {
            return true;
        }
        else if (a % 2 ==0 && b % 2 != 0) {
            return false;
        }
        else {
            return a < b;
        }
    };
    for (unsigned int i=0; i<odd_even_sorted.size()-1; i++) {
        if (f1(odd_even_sorted[i+1],odd_even_sorted[i])) {
            q3=false;
            break;
        }
    }
    if (q3) {
        std::cout<<"Q3 sorting successful!\n";
    }
    else {
        std::cout<<"Q3 sorting failed!\n";
    }
    std::cout<<"\n";
    // TODO: Verify your Q4c implementation

    // Q4d test
    std::list<SparseMatrixCoordinate<int>> sparse = {
            SparseMatrixCoordinate<int>(2, 5, 1),
            SparseMatrixCoordinate<int>(2, 2, 2),
            SparseMatrixCoordinate<int>(3, 4, 3)};

    auto f2=[](SparseMatrixCoordinate<int> a, SparseMatrixCoordinate<int> b) {
        if (a.row < b.row) {
            return true;
        }
        else if (a.row == b.row && a.col<b.col) {
            return true;
        }
        else {
            return false;
        }
    };
    sparse_matrix_sort(sparse);
    bool q4=true;
    for (std::list<SparseMatrixCoordinate<int>>::iterator it=sparse.begin(); std::next(it,1) != sparse.end(); it++) {
        if (f2(*(std::next(it,1)), *(it))) {
            q4=false;
        }
    }
    if(q4) {
        std::cout<<"Q4 sorting successful!\n";
    }
    else {
        std::cout<<"Q4 sorting failed!\n";
    }

    // TODO: Verify your Q4d implementation

    return 0;
}
