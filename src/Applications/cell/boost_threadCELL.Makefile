
boostSrcRoot=/home/vencax/cell/boost-trunk
srcdir = libs/thread/src/pthread
srcroot = ../..

binUtilsPath=/opt/cell/toolchain/bin
ROOT=/opt/cell/sysroot

CXX=$(binUtilsPath)/ppu-g++
SPUComp=$(binUtilsPath)/spu-g++
AR=$(binUtilsPath)/ppu-ar
SPU_AR=$(binUtilsPath)/spu-ar

RM=rm -f
MKDIR=mkdir

########################################
#Name of this project part
NAME=		boost_threadCELL
#Names of target sources
TARGETS=	main
########################################

#Name of target
OUTPUTNAME=	lib$(NAME).a
OUTPUTDIR=	$(srcroot)/lib
OUTPUT=		$(OUTPUTDIR)/$(OUTPUTNAME)
TMP_DIR=	$(srcroot)/tmp/$(NAME)

CXXDEBUG_OPTIONS= -ggdb 

CXXFLAGS= -Wall -Wno-deprecated $(CXXDEBUG_OPTIONS) -DBOOST_THREAD_BUILD_LIB=1
#Dependecy file creation parameters
DEP_FILE=	$(TMP_DIR)/depend
CDEPFLAGS=	-MM 

LDFLAGS=
ARFLAGS=	-r

INCLUDES=	-I$(ROOT)/usr/include

DEBUG_DEFS=	-DDEBUG_LEVEL=10\
		-DDEBUG_ADITIONAL_INFO

DEFS=		$(DEBUG_DEFS)
libSrcPath=$(boostSrcRoot)/$(srcdir)

.PHONY: all
all:		tmpdir $(OUTPUT)
		

.PHONY: tmpdir
tmpdir:		
		$(MKDIR) $(TMP_DIR) 2>/dev/null || true

.PHONY: build
build:		cleanall all

		
$(OUTPUT): exceptions.o once.o thread.o
		$(AR) $(ARFLAGS) $(OUTPUT) $(TMP_DIR)/exceptions.o $(TMP_DIR)/once.o $(TMP_DIR)/thread.o

exceptions.o:
	$(CXX) $(CXXFLAGS) $(DEFS) $(INCLUDES) -c -o $(TMP_DIR)/exceptions.o $(libSrcPath)/exceptions.cpp
once.o:
	$(CXX) $(CXXFLAGS) $(DEFS) $(INCLUDES) -c -o $(TMP_DIR)/once.o $(libSrcPath)/once.cpp
thread.o:
	$(CXX) $(CXXFLAGS) $(DEFS) $(INCLUDES) -c -o $(TMP_DIR)/thread.o $(libSrcPath)/thread.cpp

.PHONY: clean
clean:
		$(RM) $(TMP_DIR)/exceptions.o $(TMP_DIR)/once.o $(TMP_DIR)/thread.o
		$(RM) $(OUTPUT)
