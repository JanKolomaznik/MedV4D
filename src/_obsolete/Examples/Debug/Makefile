
CXX=g++
LD=ld

CXXFLAGS= -Wall -D DEBUG_LEVEL=3 -D DEBUG_ADITIONAL_INFO

INCLUDES= ../../ ../../include
CXXINCLUDES= -I$(INCLUDES)
COMMON_SRC= ../../common
OUTPUT=./main

OBJECTFILES =	\
		./Debug.o \
		./main.o

.PHONY: all
.PHONY: clean

all: 	$(OBJECTFILES)
	$(CXX) $(CXXFLAGS) $(OBJECTFILES) -o $(OUTPUT)

./main.o:	./main.cpp $(INCLUDES)/Debug.h
		$(CXX) $(CXXFLAGS) $(CXXINCLUDES) -c $< -o $@

./Debug.o:	$(COMMON_SRC)/Debug.cpp
		$(CXX) $(CXXFLAGS) $(CXXINCLUDES) -c $< -o $@

clean:
	rm -f $(OBJECTFILES) $(OUTPUT)
