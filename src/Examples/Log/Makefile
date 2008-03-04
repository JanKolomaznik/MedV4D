
CXX=g++
LD=ld

CXXFLAGS= -Wall -D DEBUG_LEVEL=3 -D DEBUG_ADITIONAL_INFO

INCLUDES= ../../include
CXXINCLUDES= -I$(INCLUDES)
COMMON_SRC= ../../common
OUTPUT=./main

OBJECTFILES =	\
		./Log.o \
		./main.o

.PHONY: all
.PHONY: clean

all: 	$(OBJECTFILES)
	$(CXX) $(CXXFLAGS) $(OBJECTFILES) -o $(OUTPUT)

./main.o:	./main.cpp $(INCLUDES)/Log.h
		$(CXX) $(CXXFLAGS) $(CXXINCLUDES) -c $< -o $@

./Log.o:	$(COMMON_SRC)/Log.cpp
		$(CXX) $(CXXFLAGS) $(CXXINCLUDES) -c $< -o $@

clean:
	rm -f $(OBJECTFILES) $(OUTPUT)
