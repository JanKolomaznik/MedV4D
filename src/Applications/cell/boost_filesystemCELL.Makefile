
boostSrcRoot=/home/vencax/cell/boost-trunk
srcdir = libs/filesystem/src
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
NAME=		boost_filesystemCELL
#Names of target sources
TARGETS=	main
########################################

#Name of target
OUTPUTNAME=	lib$(NAME).a
OUTPUTDIR=	$(srcroot)/lib
OUTPUT=		$(OUTPUTDIR)/$(OUTPUTNAME)
TMP_DIR=	$(srcroot)/tmp/$(NAME)

CXXDEBUG_OPTIONS= -g 

CXXFLAGS= -Wall -Wno-deprecated $(CXXDEBUG_OPTIONS)
#Dependecy file creation parameters
DEP_FILE=	$(TMP_DIR)/depend
CDEPFLAGS=	-MM 

LDFLAGS=
ARFLAGS=	-r

INCLUDES=	-I$(srcdir)\
		-I$(srcroot)/include\
		-I/usr/local/include/boost-1_38\
		-I$(boostSrcRoot)

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

		
$(OUTPUT): operations.o path.o portability.o utf8_codecvt_facet.o
		$(AR) $(ARFLAGS) $(OUTPUT) $(TMP_DIR)/operations.o $(TMP_DIR)/path.o $(TMP_DIR)/portability.o $(TMP_DIR)/utf8_codecvt_facet.o

operations.o:
	$(CXX) $(CXXFLAGS) $(DEFS) $(INCLUDES) -c -o $(TMP_DIR)/operations.o $(libSrcPath)/operations.cpp
path.o:
	$(CXX) $(CXXFLAGS) $(DEFS) $(INCLUDES) -c -o $(TMP_DIR)/path.o $(libSrcPath)/path.cpp
portability.o:
	$(CXX) $(CXXFLAGS) $(DEFS) $(INCLUDES) -c -o $(TMP_DIR)/portability.o $(libSrcPath)/portability.cpp
utf8_codecvt_facet.o:
	$(CXX) $(CXXFLAGS) $(DEFS) $(INCLUDES) -c -o $(TMP_DIR)/utf8_codecvt_facet.o $(libSrcPath)/utf8_codecvt_facet.cpp

.PHONY: clean
clean:
		$(RM) $(TMP_DIR)/operations.o $(TMP_DIR)/path.o $(TMP_DIR)/portability.o $(TMP_DIR)/utf8_codecvt_facet.o
		$(RM) $(OUTPUT)
