
srcdir = .
srcroot = ../../..

ITKIncludeRoot=/usr/local/include/InsightToolkit
ITKLibsRoot=/usr/local/lib/InsightToolkit
   
ITKIncludes=	-I$(ITKIncludeRoot)\
				-I$(ITKIncludeRoot)/Common\
				-I$(ITKIncludeRoot)/Algorithms\
				-I$(ITKIncludeRoot)/BasicFilters\
				-I$(ITKIncludeRoot)/IO\
				-I$(ITKIncludeRoot)/Utilities\
				-I$(ITKIncludeRoot)/Utilities/vxl/vcl\
				-I$(ITKIncludeRoot)/Utilities/vxl/core\
				-I$(ITKIncludeRoot)/gdcm/src

ifdef COMPILE_FOR_CELL
############################### CellBE
binUtilsPath=/opt/cell/toolchain/bin
ROOT=/opt/cell/sysroot

CXX=$(binUtilsPath)/ppu-g++
SPUComp=$(binUtilsPath)/spu-g++
AR=$(binUtilsPath)/ppu-ar
SPU_AR=$(binUtilsPath)/spu-ar
archPostfix=CELL

INCLUDES=	-I$(ROOT)/usr/include\
			-I$(binUtilsPath)/lib/gcc/ppu/4.1.1/include\
			-I$(binUtilsPath)/lib/gcc/ppu/4.1.1/include/c++\
			#-I$(srcdir)\
			#-I$(srcroot)/include\
			#-I$(srcroot)\
			$(ITKIncludes)

LIBDIRS=	-L$(ROOT)/lib64\
			-L/usr/local/lib/InsightToolkit

#M4DLIBS=	$(srcroot)/lib/libImaging$(archPostfix).a\
#		$(srcroot)/lib/libcommon$(archPostfix).a

LIBS=	-l$(ITKLIBS)\
		#$(M4DLIBS)\		
		#$(srcroot)/lib/libboost_threadCELL.a \
		#$(srcroot)/lib/libboost_filesystemCELL.a\
		#$(srcroot)/lib/libfileStreamCELL.a\
		#-lpthread
		#$(ROOT)/usr/lib64/libpthread_nonshared.a
CXXDEBUG_OPTIONS= -g
else
##################### NORMAL PC
CXX=g++
AR=ar
ROOT=
archPostfix=
LIBDIRS=	-L$(srcroot)/lib\
			-L/usr/local/lib

INCLUDES=	-I$(srcdir)\
			-I$(srcroot)/include\
			-I$(srcroot)\
			-I/usr/local/include/boost-1_38\
			$(ITKIncludes)
			
M4DLIBS=	$(srcroot)/lib/libImaging.a\
		$(srcroot)/lib/libcommon.a\
		$(srcroot)/lib/libfileStream.a

ITKLIBS= 	$(ITKLibsRoot)/libITKNumerics.a\
			$(ITKLibsRoot)/libITKIO.a\
			$(ITKLibsRoot)/libITKNrrdIO.a\
			$(ITKLibsRoot)/libitkgdcm.a\
			$(ITKLibsRoot)/libitkjpeg12.a\
			$(ITKLibsRoot)/libitkjpeg16.a\
			$(ITKLibsRoot)/libitkopenjpeg.a\
			$(ITKLibsRoot)/libitkpng.a\
			$(ITKLibsRoot)/libitktiff.a\
			$(ITKLibsRoot)/libitkjpeg8.a\
			$(ITKLibsRoot)/libITKSpatialObject.a\
			$(ITKLibsRoot)/libITKCommon.a\
			$(ITKLibsRoot)/libitkvnl_inst.a\
			$(ITKLibsRoot)/libitkvnl_algo.a\
			$(ITKLibsRoot)/libitkvnl.a\
			$(ITKLibsRoot)/libitkv3p_netlib.a\
			$(ITKLibsRoot)/libitkvcl.a\
			$(ITKLibsRoot)/libITKMetaIO.a\
			$(ITKLibsRoot)/libitksys.a\
			$(ITKLibsRoot)/libITKDICOMParser.a\
			$(ITKLibsRoot)/libITKEXPAT.a\
			$(ITKLibsRoot)/libITKniftiio.a\
			$(ITKLibsRoot)/libITKznz.a\
			$(ITKLibsRoot)/libitkzlib.a
			

LIBS=	-lm -lpthread -ldl $(ITKLIBS)		
		#-lboost_thread\
		#-lboost_filesystem
		
CXXDEBUG_OPTIONS= -ggdb
endif

RM=rm -f
MKDIR=mkdir

########################################
#Name of this project part
NAME=		loadDICOMSeries$(archPostfix)
#Names of target sources
TARGETS=	main
########################################

#Name of target
OUTPUTNAME=	$(NAME)
OUTPUTDIR=	$(srcroot)/executables
OUTPUT=		$(OUTPUTDIR)/$(OUTPUTNAME)
#TMP_DIR=	$(srcroot)/tmp/$(NAME)
TMP_DIR=	Build

CXXFLAGS= -Wall -Wno-deprecated $(CXXDEBUG_OPTIONS)
#Dependecy file creation parameters
#DEP_FILE=	$(TMP_DIR)/depend
CDEPFLAGS=	-MM 

LDFLAGS=
ARFLAGS=	-r

DEBUG_DEFS=	-DDEBUG_LEVEL=10\
		-DDEBUG_ADITIONAL_INFO

DEFS=		$(DEBUG_DEFS)

.PHONY: all
all:		tmpdir $(OUTPUT)
		

.PHONY: tmpdir
tmpdir:		
		$(MKDIR) $(TMP_DIR) 2>/dev/null || true

.PHONY: build
build:		cleanall all

.PHONY: dep
dep:		
		$(CXX) $(INCLUDES) $(CDEPFLAGS) $(SRC_FILES) | \
#		sed 's/\(.*\)\.o:/$$(TMP_DIR)\/\1\.o:/' > $(DEP_FILE)

OBJ_FILES=	$(TMP_DIR)/main.o\
			$(TMP_DIR)/segmentator.o

$(OUTPUT): $(OBJ_FILES)
		$(CXX) $(CXXFLAGS) $(LIBDIRS) $(OBJ_FILES) $(LIBS) -o $(TMP_DIR)/main

$(TMP_DIR)/main.o:
		$(CXX) $(CXXFLAGS) $(DEFS) $(INCLUDES) -o $(TMP_DIR)/main.o -c main.cpp
		
$(TMP_DIR)/segmentator.o:
		$(CXX) $(CXXFLAGS) $(DEFS) $(INCLUDES) -o $(TMP_DIR)/segmentator.o -c segmentator.cpp

.PHONY: clean
clean:
		$(RM) $(OBJ_FILES) $(DEP_FILE)
		$(RM) $(OUTPUT)
		#rmdir $(TMP_DIR)