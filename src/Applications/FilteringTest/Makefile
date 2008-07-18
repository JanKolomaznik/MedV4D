
srcdir = .

CXX=g++
AR=ar
RM=rm -f
MKDIR=mkdir
MOC=moc
RCC=rcc


########################################
#Name of this project part
NAME=		FilteringTest
#Names of target sources
TARGETS=	main

M4DLIBS=	../../lib/libm4dWidgets.a\
		../../lib/libImaging.a\
		../../lib/libvtkIntegration.a\
		../../lib/libbackendForDICOM.a\
		../../lib/libcommon.a

QTLIBS=		-lQtCore\
		-lQtGui
VTKLIBS=	-lvtkCommon\
		-lvtkDICOMParser\
		-lvtkexoIIc\
		-lvtkFiltering\
		-lvtkftgl\
		-lvtkGenericFiltering\
		-lvtkGraphics\
		-lvtkHybrid\
		-lvtkImaging\
		-lvtkIO\
		-lvtkNetCDF\
		-lvtkRendering\
		-lvtksys\
		-lvtkVolumeRendering\
		-lvtkWidgets\
		-lQVTK\
		-lwrap

DCMTKLIBS=	-lofstd\
		-ldcmdata\
		-ldcmnet

LIBS=		$(M4DLIBS)\
		$(QTLIBS)\
		$(VTKLIBS)\
		$(DCMTKLIBS)\
		-lboost_thread
########################################

#Name of target
OUTPUTNAME=	$(NAME)
OUTPUTDIR=	../../executables
OUTPUT=		$(OUTPUTDIR)/$(OUTPUTNAME)
TMP_DIR=	../../tmp/$(NAME)

CXXDEBUG_OPTIONS= -ggdb

CXXFLAGS= -Wall -Wno-deprecated $(CXXDEBUG_OPTIONS)
#Dependecy file creation parameters
DEP_FILE=	$(TMP_DIR)/depend
CDEPFLAGS=	-MM 

LDFLAGS=
ARFLAGS=	-r
LIBDIRS=	-L../../lib\
		-L/usr/lib/qt4
INCLUDES=	-I$(srcdir)\
		-I../../include

DEBUG_DEFS=	-DDEBUG_LEVEL=10\
		-DDEBUG_ADITIONAL_INFO

DEFS=		$(DEBUG_DEFS)


SRC_FILES=	$(addsuffix .cpp, $(TARGETS) ) 
OBJ_FILES=	$(addprefix $(TMP_DIR)/, $(addsuffix .o, $(TARGETS) ) )


.PHONY: all
all:		tmpdir dep $(OUTPUT)
		

.PHONY: tmpdir
tmpdir:		
		$(MKDIR) $(TMP_DIR) 2>/dev/null || true

.PHONY: build
build:		cleanall all

.PHONY: dep
dep:		
		$(CXX) $(INCLUDES) $(CDEPFLAGS) $(SRC_FILES) | \
		sed 's/\(.*\)\.o:/$$(TMP_DIR)\/\1\.o:/' > $(DEP_FILE)

$(TMP_DIR)/%.o:	%.cpp	
		$(CXX) $(CXXFLAGS) $(DEFS) $(INCLUDES) -c $< -o $@

-include $(DEP_FILE)
		
$(OUTPUT):	$(OBJ_FILES)
		$(CXX) $(CXXFLAGS) $(OBJ_FILES) $(LIBDIRS) $(LIBS) -o $(OUTPUT) 

.PHONY: clean
clean:
		$(RM) $(OBJ_FILES) $(DEP_FILE)
		$(RM) $(OUTPUT)
