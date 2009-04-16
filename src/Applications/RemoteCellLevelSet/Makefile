
srcTop = ../..
srcdir = .

CXX=g++
AR=ar
RM=rm -f
MKDIR=mkdir
MOC=moc
RCC=rcc

include $(srcTop)/project/pathsDefinitions.mk

########################################
#Name of this project part
NAME=		RemoteCellLevelSet
#Names of target sources
TARGETS=	main \
		mainWindow \
		SettingsBox

MOCINPUT=	mainWindow.h \
		SettingsBox.h

RCCINPUT=	mainWindow.qrc

M4DLIBS=	$(srcTop)/lib/libm4dWidgets.a\
			$(srcTop)/lib/libvtkIntegration.a\
			$(srcTop)/lib/libbackendForDICOM.a\
			$(srcTop)/lib/libImaging.a\
			$(srcTop)/lib/libcommon.a\
			$(srcTop)/lib/libremoteCompCommons.a\
			$(srcTop)/lib/libremoteClient.a

QTLIBS=		-lQtCore\
		-lQtGui\
		-lQtOpenGL

LIBS=		$(M4DLIBS)\
		$(QTLIBS)\
		$(VTKLIBS)\
		$(DCMTKLIBS)\
		-lboost_thread-mt \
		-lboost_filesystem \
		-lGLU
########################################

#Name of target
OUTPUTNAME=	$(NAME)
OUTPUTDIR=	$(srcTop)/executables
OUTPUT=		$(OUTPUTDIR)/$(OUTPUTNAME)
TMP_DIR=	$(srcTop)/tmp/$(NAME)

CXXDEBUG_OPTIONS= -ggdb 

CXXFLAGS= -Wall -Wno-deprecated $(CXXDEBUG_OPTIONS)
#Dependecy file creation parameters
DEP_FILE=	$(TMP_DIR)/depend
CDEPFLAGS=	-MM 

LDFLAGS=
ARFLAGS=	-r
LIBDIRS=	-L$(srcTop)/lib\
		-L/usr/lib/qt4\
		-L$(DCMTK_LIB_PATH)\
		-L$(VTKLibsDir)
		
QTINCLUDES = \
		-I/usr/include/qt4\
		-I/usr/include/qt4/Qt\
		-I/usr/include/qt4/QtCore\
		-I/usr/include/qt4/QtGui\
		-I/usr/include/Qt\
		-I/usr/include/QtGui\
		-I/usr/include/QtCore\
		
INCLUDES=	-I$(srcTop) \
		-I$(srcdir)\
		-I$(srcTop)/include\
		-I$(VTKIncludeDir)\
		$(QTINCLUDES)\
		$(ITKIncludes)


MOCOUTPUT=      $(addprefix $(TMP_DIR)/moc_, $(MOCINPUT:.h=.cpp) )

DEBUG_DEFS=	-DDEBUG_LEVEL=10\
		-DDEBUG_ADITIONAL_INFO

DEFS=		$(DEBUG_DEFS)


SRC_FILES=	$(addsuffix .cpp, $(TARGETS) ) 
OBJ_FILES=	$(addprefix $(TMP_DIR)/,   $(addprefix rcc_, $(RCCINPUT:.qrc=.o ) ) $(addprefix moc_, $(MOCINPUT:.h=.o) ) $(addsuffix .o, $(TARGETS) ) )


.PHONY: all
all:		tmpdir Common Imaging DICOM VTKInt GUI commonRCPart clientRCPart dep $(OUTPUT)

commonRCPart:
	$(MAKE) --directory=$(srcTop)/remoteComp/commonPart
	
clientRCPart:
	$(MAKE) --directory=$(srcTop)/remoteComp/clientPart
	
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

commonRCPartCLEAN:
	$(MAKE) --directory=$(srcTop)/remoteComp/commonPart clean
	
clientRCPartCLEAN:
	$(MAKE) --directory=$(srcTop)/remoteComp/clientPart clean
	
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
clean: commonRCPartCLEAN clientRCPartCLEAN #GUICLEAN DICOMCLEAN ImagingCLEAN CommonCLEAN
		$(RM) $(OBJ_FILES) $(MOCOUTPUT) $(addprefix $(TMP_DIR)/rcc_, $(RCCINPUT:.qrc=.cpp ) ) $(DEP_FILE)
		$(RM) $(OUTPUT)
