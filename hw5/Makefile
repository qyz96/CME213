MPICXX=mpic++
MPICXXFLAGS=-O3 -std=c++11 -Wall -Wextra -fopenmp

CXX=g++
CXXFLAGS=-O3 -std=c++11 -Wall -Wextra -fopenmp 

all: dnsmmm dnsmmm_timing serialmmm

dnsmmm: dnsmmm.cpp dns.h
	$(MPICXX) -o $@ $(MPICXXFLAGS) dnsmmm.cpp

dnsmmm_timing: dnsmmm.cpp dns.h
	$(MPICXX) -o $@ $(MPICXXFLAGS) -DTIMING dnsmmm.cpp 

serialmmm: serialmmm.cpp
	 $(CXX) -o $@ $(CXXFLAGS) $^
	 
clean:
	rm -f *.o dnsmmm dnsmmm_timing serialmmm 
