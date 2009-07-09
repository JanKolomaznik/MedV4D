srcTop = ../..
srcdir = .

include $(srcTop)/project/ToolchainSelect.mk
include $(srcTop)/project/pathsDefinitions.mk
include $(srcTop)/project/configurationSelector.mk


########################################
#Name of this project part
NAME=		OrganSegmentation
#Names of target sources
TARGETS=	main \
		mainWindow \
		SegmentationWidget \
		ManualSegmentationManager \
		KidneySegmentationManager \
		MainManager \
		SettingsBox

MOCINPUT=	mainWindow.h \
		SegmentationWidget.h \
		SettingsBox.h \
		ManualSegmentationManager.h \
		MainManager.h \
		KidneySegmentationManager.h

RCCINPUT=	mainWindow.qrc
########################################

#Name of target
OUTPUTNAME=	$(NAME)
OUTPUTDIR=	../../executables/$(CONF)
OUTPUT=		$(OUTPUTDIR)/$(OUTPUTNAME)
TMP_DIR=	../../tmp/$(CONF)/$(NAME)

M4DLIBS=	-lm4dWidgets\
		-lvtkIntegration\
		-lbackendForDICOM\
		-lImaging\
		-lcommon

LIBS=		$(M4DLIBS)\
		$(DCMTKLIBS)\
		$(VTKLIBS)\
		$(QTLIBS)\
		-lboost_thread-mt \
		-lboost_filesystem-mt \
		-lGLU

LIBDIRS=	-L$(srcTop)/lib/$(CONF)\
		$(QTLibDirs)\
			-L$(DCMTK_LIB_PATH)\
			-L$(VTKLibsDir)

INCLUDES= -I$(srcTop)\
		  -I$(srcdir)\
		  -I$(VTKIncludeDir)\
			$(QTIncludDirs)\
			$(ITKIncludes)


MOCOUTPUT=      $(addprefix $(TMP_DIR)/moc_, $(MOCINPUT:.h=.cpp) )


SRC_FILES=	$(addsuffix .cpp, $(TARGETS) )
OBJ_FILES=	$(addprefix $(TMP_DIR)/,   $(addprefix rcc_, $(RCCINPUT:.qrc=.o ) ) $(addprefix moc_, $(MOCINPUT:.h=.o) ) $(addsuffix .o, $(TARGETS) ) )


.PHONY: all
all:		outDir tmpdir dep Common Imaging DICOM VTKInt GUI $(OUTPUT)


GUI:
	$(MAKE) --directory=$(srcTop)/GUI/widgets all

VTKInt:
	$(MAKE) --directory=$(srcTop)/vtkIntegration all

DICOM:
	$(MAKE) --directory=$(srcTop)/backendForDICOM all

Imaging:
	$(MAKE) --directory=$(srcTop)/Imaging all

Common:
	$(MAKE) --directory=$(srcTop)/common all

## CLEANING ##################################

GUICLEAN:
	$(MAKE) --directory=$(srcTop)/GUI/widgets clean

DICOMCLEAN:
	$(MAKE) --directory=$(srcTop)/backendForDICOM clean

ImagingCLEAN:
	$(MAKE) --directory=$(srcTop)/Imaging clean

CommonCLEAN:
	$(MAKE) --directory=$(srcTop)/common clean

##############################################

.PHONY: build
build:		cleanall all

.PHONY: dep
dep:
		$(CXX) $(INCLUDES) $(CDEPFLAGS) $(SRC_FILES) | \
		sed 's/\(.*\)\.o:/$$(TMP_DIR)\/\1\.o:/' > $(DEP_FILE)

$(TMP_DIR)/moc_%.cpp:   %.h
		$(MOC) $< -o $@

$(TMP_DIR)/rcc_%.cpp:   $(RCCINPUT)
		$(RCC) -name $(*F) $< -o $@

$(TMP_DIR)/%.o:	%.cpp
		$(CXX) $(CXXFLAGS) $(DEFS) $(INCLUDES) -c $< -o $@

$(TMP_DIR)/%.o:	$(TMP_DIR)/%.cpp
		$(CXX) $(CXXFLAGS) $(DEFS) $(INCLUDES) -c $< -o $@

-include $(DEP_FILE)

$(OUTPUT):	$(OBJ_FILES)
		$(CXX) $(CXXFLAGS) $(OBJ_FILES) $(LIBDIRS) $(LIBS) -o $(OUTPUT)

.PHONY: clean
clean:	
		$(RM) $(OBJ_FILES) $(MOCOUTPUT) $(addprefix $(TMP_DIR)/rcc_, $(RCCINPUT:.qrc=.cpp ) ) $(DEP_FILE)
		$(RM) $(OUTPUT)
