srcTop = ../..
srcdir = .

include $(srcTop)/project/ToolchainSelect.mk
include $(srcTop)/project/pathsDefinitions.mk
include $(srcTop)/project/configurationSelector.mk


########################################
#Name of this project part
NAME=		m4dPilot
#Names of target sources
TARGETS=	main\
		mainWindow

MOCINPUT=	mainWindow.h

RCCINPUT=	mainWindow.qrc

M4DLIBS=	../../lib/libm4dWidgets.a\
		../../lib/libvtkIntegration.a\
		../../lib/libbackendForDICOM.a\
		../../lib/libImaging.a\
		../../lib/libcommon.a

LIBS=		$(M4DLIBS)\
		$(DCMTKLIBS)\
		$(VTKLIBS)\
		$(QTLIBS)\
		-lboost_thread-mt \
		-lboost_filesystem \
		-lGLU
########################################

#Name of target
OUTPUTNAME=	$(NAME)
OUTPUTDIR=	../../executables
OUTPUT=		$(OUTPUTDIR)/$(OUTPUTNAME)
TMP_DIR=	../../tmp/$(NAME)

LIBDIRS=	-L$(srcTop)/lib\
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
all:		tmpdir dep Common Imaging DICOM VTKInt GUI $(OUTPUT)

	
GUI:
	$(MAKE) --directory=$(srcTop)/GUI/widgets
	
VTKInt:
	$(MAKE) --directory=$(srcTop)/vtkIntegration
	
DICOM:
	$(MAKE) --directory=$(srcTop)/backendForDICOM
	
Imaging:
	$(MAKE) --directory=$(srcTop)/Imaging
	
Common:
	$(MAKE) --directory=$(srcTop)/common
	
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
		

.PHONY: tmpdir
tmpdir:		
		$(MKDIR) $(TMP_DIR) 2>/dev/null || true

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
		
$(OUTPUT):	$(OBJ_FILES) $(M4DLIBS)
		$(CXX) $(CXXFLAGS) $(OBJ_FILES) $(LIBDIRS) $(LIBS) -o $(OUTPUT) 

.PHONY: clean
clean:	DICOMCLEAN #ImagingCLEAN CommonCLEAN GUICLEAN 
		$(RM) $(OBJ_FILES) $(MOCOUTPUT) $(addprefix $(TMP_DIR)/rcc_, $(RCCINPUT:.qrc=.cpp ) ) $(DEP_FILE)
		$(RM) $(OUTPUT)
