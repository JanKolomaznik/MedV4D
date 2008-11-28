
srcdir = .

CXX=g++
AR=ar
RM=rm -f
MKDIR=mkdir
MOC=moc
RCC=rcc


########################################
#Name of this project part
NAME=		tools
#Names of target sources
TARGETS=	DICOM2ImageDump \
		ImageDumpView \
		GaussianFilter

M4DLIBS=	../../lib/libm4dWidgets.a\
		../../lib/libImaging.a\
		../../lib/libvtkIntegration.a\
		../../lib/libbackendForDICOM.a\
		../../lib/libcommon.a

QTLIBS=		-lQtCore\
		-lQtGui\
		-lQtOpenGL

DCMTKLIBS=	-lofstd\
		-ldcmdata\
		-ldcmnet

OTHERLIBS=	-lz\
		-lwrap\
		-lboost_thread \
		-lboost_filesystem \
		-lGLU

LIBS=		$(M4DLIBS)\
		$(QTLIBS)\
		$(VTKLIBS)\
		$(DCMTKLIBS)\
		$(OTHERLIBS)

########################################


OUTPUTDIR=	../../executables/tools
TMP_DIR=		../../tmp/tools

OUTPUTS= $(addprefix $(OUTPUTDIR)/, $(TARGETS) )

CXXDEBUG_OPTIONS= -ggdb 

CXXFLAGS= -Wall -Wno-deprecated $(CXXDEBUG_OPTIONS)
#Dependecy file creation parameters
DEP_FILE=	$(TMP_DIR)/depend
CDEPFLAGS=	-MM 

LDFLAGS=
ARFLAGS=	-r
LIBDIRS=	-L../../lib
INCLUDES=	-I../.. \
		-I$(srcdir)\
		-I../../include\
		-I/usr/include/vtk-5.0\
		-I/usr/include/qt4\
		-I/usr/include/qt4/Qt\
		-I/usr/include/qt4/QtCore\
		-I/usr/include/qt4/QtGui

DEBUG_DEFS=	-DDEBUG_LEVEL=10\
		-DDEBUG_ADITIONAL_INFO

DEFS=		$(DEBUG_DEFS)

.PHONY: all
all:		tmpdir $(OUTPUTS)

.PHONY: tmpdir
tmpdir:         
		$(MKDIR) $(TMP_DIR) 2>/dev/null || true
		$(MKDIR) $(OUTPUTDIR) 2>/dev/null || true

.PHONY: build
build:		cleanall all

.PHONY: dep
dep:		
		$(CXX) $(INCLUDES) $(CDEPFLAGS) $(SRC_FILES) | \
		sed 's/\(.*\)\.o:/$$(TMP_DIR)\/\1\.o:/' > $(DEP_FILE)

-include $(DEP_FILE)
		
$(TMP_DIR)/%.o: %.cpp   
		$(CXX) $(CXXFLAGS) $(DEFS) $(INCLUDES) -c $< -o $@


$(OUTPUTDIR)/%:	$(TMP_DIR)/%.o $(M4DLIBS) 
		$(CXX) $(CXXFLAGS) $< $(LIBDIRS) $(LIBS) -o $@

.PHONY: clean
clean:
		$(RM) $(OUTPUTS)
		$(RM) $(TMP_DIR)/*

