srcTop = ..
srcdir = .

CXX=g++
AR=ar
RM=rm -f
MKDIR=mkdir

########################################
#Name of this project part
NAME=		backendForDICOM
#Names of target sources
TARGETS=	AbstractService \
			DicomAssoc \
			DicomObj \
			DICOMServiceProvider \
			FindService \
			LocalService \
			MoveService \
			StoreService 
########################################

#Name of target
OUTPUTNAME=	lib$(NAME).a
OUTPUTDIR=	$(srcTop)/lib
OUTPUT=		$(OUTPUTDIR)/$(OUTPUTNAME)
TMP_DIR=	$(srcTop)/tmp/$(NAME)
SRC_DIR=	./src

CXXDEBUG_OPTIONS= -ggdb

CXXFLAGS= -Wall -Wno-deprecated $(CXXDEBUG_OPTIONS)
#Dependecy file creation parameters
DEP_FILE=	$(TMP_DIR)/depend
CDEPFLAGS=	-MM 

LDFLAGS=
ARFLAGS=	-r
INCLUDES=	-I$(srcTop) \
		-I$(srcdir)\
		-I$(srcTop)/include\
		-I/usr/local/dicom/include


DEBUG_DEFS=	-DDEBUG_LEVEL=10\
		-DDEBUG_ADITIONAL_INFO
		

DEFS=		-DHAVE_CONFIG_H\
		-DHAVE_SSTREAM\
		$(DEBUG_DEFS)


OBJ_FILES=	$(TMP_DIR)/AbstractService.o\
			$(TMP_DIR)/DicomAssoc.o \
			$(TMP_DIR)/DicomObj.o \
			$(TMP_DIR)/DICOMServiceProvider.o \
			$(TMP_DIR)/FindService.o \
			$(TMP_DIR)/LocalService.o \
			$(TMP_DIR)/MoveService.o \
			$(TMP_DIR)/StoreService.o 
.PHONY: all
all:		tmpdir $(OUTPUT)
		

.PHONY: tmpdir
tmpdir:		
		$(MKDIR) $(TMP_DIR) 2>/dev/null || true

.PHONY: build
build:		cleanall all

$(TMP_DIR)/AbstractService.o:	
		$(CXX) $(CXXFLAGS) $(DEFS) $(INCLUDES) -o $(TMP_DIR)/AbstractService.o -c $(SRC_DIR)/AbstractService.cpp
		
$(TMP_DIR)/DicomAssoc.o:	
		$(CXX) $(CXXFLAGS) $(DEFS) $(INCLUDES) -o $(TMP_DIR)/DicomAssoc.o -c $(SRC_DIR)/DicomAssoc.cpp
		
$(TMP_DIR)/DicomObj.o:	
		$(CXX) $(CXXFLAGS) $(DEFS) $(INCLUDES) -o $(TMP_DIR)/DicomObj.o -c $(SRC_DIR)/DicomObj.cpp
		
$(TMP_DIR)/DICOMServiceProvider.o:	
		$(CXX) $(CXXFLAGS) $(DEFS) $(INCLUDES) -o $(TMP_DIR)/DICOMServiceProvider.o -c $(SRC_DIR)/DICOMServiceProvider.cpp
		
$(TMP_DIR)/FindService.o:	
		$(CXX) $(CXXFLAGS) $(DEFS) $(INCLUDES) -o $(TMP_DIR)/FindService.o -c $(SRC_DIR)/FindService.cpp
		
$(TMP_DIR)/MoveService.o:	
		$(CXX) $(CXXFLAGS) $(DEFS) $(INCLUDES) -o $(TMP_DIR)/MoveService.o -c $(SRC_DIR)/MoveService.cpp
		
$(TMP_DIR)/LocalService.o:	
		$(CXX) $(CXXFLAGS) $(DEFS) $(INCLUDES) -o $(TMP_DIR)/LocalService.o -c $(SRC_DIR)/LocalService.cpp
		
$(TMP_DIR)/StoreService.o:	
		$(CXX) $(CXXFLAGS) $(DEFS) $(INCLUDES) -o $(TMP_DIR)/StoreService.o -c $(SRC_DIR)/StoreService.cpp
		
$(OUTPUT):	$(OBJ_FILES)
		$(AR) $(ARFLAGS) $(OUTPUT) $(OBJ_FILES)

.PHONY: clean
clean:
		$(RM) $(TMP_DIR)/*.o
		$(RM) $(OUTPUT)
