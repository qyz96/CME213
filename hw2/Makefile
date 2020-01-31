CXX=/usr/local/bin/g++-9
CXXFLAGS=-std=c++11 -g -Wall -O3 -fopenmp
LDFLAGS=

# For Mac
#CXXFLAGS=-std=c++11 -g -Wall -O3 -I/usr/local/include -Xpreprocessor -fopenmp
#LDFLAGS=-L/usr/local/lib -lomp

SRC1=main_q1.cpp tests_q1.cpp
SRC2=main_q2.cpp tests_q2.cpp
INC1=tests_q1.h
INC2=test_macros.h tests_q2.h

default: main_q1 main_q2 main_q2_part6
main_q1: $(INC1) $(SRC1)
	$(CXX) $(CXXFLAGS) $(SRC1) $(LDFLAGS) -o $@
main_q2: $(INC2) $(SRC2)
	$(CXX) $(CXXFLAGS) $(SRC2) -o $@ $(LDFLAGS)
main_q2_part6: $(INC2) $(SRC2)
	$(CXX) $(CXXFLAGS) $(SRC2) -D QUESTION6 -o $@ $(LDFLAGS)

clean:
	rm -f *.o *~ *~ main_q1 main_q2 main_q2_part6
	rm -rf *.dSYM
