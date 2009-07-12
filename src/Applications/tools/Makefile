srcTop = ../..
srcdir = .

include $(srcTop)/project/ToolchainSelect.mk
include $(srcTop)/project/pathsDefinitions.mk
include $(srcTop)/project/configurationSelector.mk


########################################
#Name of this project part
NAME=		tools
#Names of target sources
TARGETS=	DICOM2ImageDump \
		ImageDumpView \
		GaussianFilter \
		MedianFilter \
		SobelEdgeDetector \
		CannyEdgeDetector \
		SobelGradientOperator \
		LaplaceOperator \
		ImageCrop \
		ClampFilter \
		CreateExampleData \
		MaxFilter \
		MinFilter

M4DLIBS=	-lm4dWidgets\
		-lvtkIntegration\
		-lbackendForDICOM\
		-lImaging\
		-lcommon

QTLIBS=		-lQtCore\
		-lQtGui\
		-lQtOpenGL

DCMTKLIBS=	-ldcmnet -ldcmdata -lofstd

OTHERLIBS=	-lz\
		-lwrap\
		-lboost_thread-mt \
		-lboost_filesystem-mt \
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

#Dependecy file creation parameters
DEP_FILE=	$(TMP_DIR)/depend

LIBDIRS=	-L$(srcTop)/lib/$(CONF)\
		-L../../lib
INCLUDES=	-I../.. \
		-I$(srcdir)\
		-I../../include\
		-I/usr/include/vtk-5.0\
		-I/usr/include/qt4\
		-I/usr/include/qt4/Qt\
		-I/usr/include/qt4/QtCore\
		-I/usr/include/qt4/QtGui


.PHONY: all
all:		tmpdir $(OUTPUTS)

.PHONY: build
build:		cleanall all

.PHONY: dep
dep:		
		$(CXX) $(INCLUDES) $(CDEPFLAGS) $(SRC_FILES) | \
		sed 's/\(.*\)\.o:/$$(TMP_DIR)\/\1\.o:/' > $(DEP_FILE)

-include $(DEP_FILE)
		
$(TMP_DIR)/%.o: %.cpp   
		$(CXX) $(CXXFLAGS) $(DEFS) $(INCLUDES) -c $< -o $@


$(OUTPUTDIR)/%:	$(TMP_DIR)/%.o 
		#$(M4DLIBS) 
		$(CXX) $(CXXFLAGS) $< $(LIBDIRS) $(LIBS) -o $@

.PHONY: clean
clean:
		$(RM) $(OUTPUTS)
		$(RM) $(TMP_DIR)/*

