CXX=nvcc
CXXFLAGS=-g -std=c++11 --compiler-options -Wall -O3 -arch=sm_37
SRC1=main_q1.cu
SRC2=main_q2.cu
SRC3=test.cu

default: main_q1 main_q2 test
main_q1: $(SRC1)
	$(CXX) $(CXXFLAGS) $(SRC1) -o $@
main_q2: $(SRC2)
	$(CXX) $(CXXFLAGS) $(SRC2) -o $@
test: $(SRC2)
	$(CXX) $(CXXFLAGS) $(SRC3) -o $@

clean:
	rm -f *.o *~ *~ main_q1 main_q2 test
	rm -rf *.dSYM
