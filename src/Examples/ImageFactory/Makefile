
CXX=g++
LD=ld

CXXFLAGS= -Wall -D DEBUG_LEVEL=20 -D DEBUG_ADITIONAL_INFO

INCLUDES= ../../include
CXXINCLUDES= -I$(INCLUDES)
COMMON_SRC= ../../common
TMP=../../tmp
IMAGING_SRC=../../Imaging

OUTPUT=./main


COMMON_OBJ_FILES = \
			$(TMP)/Log.o \
			$(TMP)/Debug.o \
			$(TMP)/ExceptionBase.o 
COMMON_SOURCE_FILES = $(patsubst $(TMP)/%.o, $(COMMON_SRC)/%.cpp,$(COMMON_OBJ_FILES))



IMAGING_OBJ_FILES = \
			$(TMP)/ImageFactory.o\
			$(TMP)/AbstractImage.o
IMAGING_SOURCE_FILES = $(patsubst $(TMP)/%.o, $(IMAGING_SRC)/%.cpp,$(IMAGING_OBJ_FILES))


OBJECTFILES =	\
		./main.o

.PHONY: all
.PHONY: clean

all: 	$(OBJECTFILES) $(COMMON_OBJ_FILES) $(IMAGING_OBJ_FILES)
	$(CXX) $(CXXFLAGS) $(OBJECTFILES) $(COMMON_OBJ_FILES) $(IMAGING_OBJ_FILES) -o $(OUTPUT)

./main.o:	./main.cpp
		$(CXX) $(CXXFLAGS) $(CXXINCLUDES) -c $< -o $@

$(COMMON_OBJ_FILES): $(COMMON_SOURCE_FILES)
		$(CXX) $(CXXFLAGS) $(CXXINCLUDES) -c $(patsubst $(TMP)/%.o, $(COMMON_SRC)/%.cpp,$@) -o $@

$(IMAGING_OBJ_FILES): $(IMAGING_SOURCE_FILES)
		$(CXX) $(CXXFLAGS) $(CXXINCLUDES) -c $(patsubst $(TMP)/%.o, $(IMAGING_SRC)/%.cpp,$@) -o $@

clean:
	rm -f $(OBJECTFILES)  $(COMMON_OBJ_FILES) $(IMAGING_OBJ_FILES) $(OUTPUT)

