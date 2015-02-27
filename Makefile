# Define EIGEN and ADEPT in your Makefile.include. 
# Example:
#  EIGEN=-I/home/software/eigen
#  ADEPT=-I/home/software/adept-1.0/include -L-I/home/software/adept-1.0/lib
include Makefile.include

all: simple-nn

simple-nn: simple-nn.cc
	g++ $(EIGEN) $(ADEPT) -msse3 -march=native -O3 -o $@ $< -std=c++11 -Wall -ladept

clean:
	rm -rf simple-nn simple-nn.dSYM

