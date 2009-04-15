srcTop = ../..
srcdir = .

CC=g++
AR=ar
RM=rm -f
MKDIR=mkdir
MOC=moc
RCC=rcc

include $(srcTop)/project/pathsDefinitions.mk

########################################
#Name of this project part
NAME=		m4dWidgets
#Names of target sources
TARGETS=	m4dGUIMainWindow \
		m4dGUIMainWindow2 \
		m4dGUIMainViewerDesktopWidget \
		m4dGUIVtkViewerWidget \
		m4dGUIStudyManagerWidget \
		m4dGUIScreenLayoutWidget \
		m4dGUISliceViewerWidget \
		m4dGUISliceViewerWidget2 \
		m4dGUIToolBarCustomizerWidget\
		m4dGUIProgressBarWidget

SUBDIR1=	components/src

SUBDIR2=	ogl/src

SUBDIR3=	utils/src

TARGETS1=	StManagerFilterComp \
		StManagerHProtocolsComp \
		StManagerStudyListComp
		
TARGETS2=	fonts

TARGETS3=	LoadingThreads \
		StudyFilter 

MOCINPUT=	m4dGUIAbstractViewerWidget.h \
		m4dGUIMainWindow.h \
		m4dGUIMainWindow2.h \
		m4dGUIMainViewerDesktopWidget.h \
		m4dGUIScreenLayoutWidget.h \
		m4dGUIStudyManagerWidget.h \
		m4dGUIVtkViewerWidget.h \
		m4dGUISliceViewerWidget.h \
		m4dGUISliceViewerWidget2.h \
		m4dGUIToolBarCustomizerWidget.h \
		StManagerFilterComp.h \
		StManagerHProtocolsComp.h \
		StManagerStudyListComp.h \
		m4dGUIProgressBarWidget.h \
		LoadingThreads.h


RCCINPUT=	m4dGUIMainWindow.qrc \
		m4dGUIMainWindow2.qrc \
		m4dGUIStudyManagerWidget.qrc \
		m4dGUIScreenLayoutWidget.qrc \
		m4dGUIToolBarCustomizerWidget.qrc
########################################

#Name of target
OUTPUTNAME=	lib$(NAME).a
OUTPUTDIR=	../../lib
OUTPUT=		$(OUTPUTDIR)/$(OUTPUTNAME)
TMP_DIR=	../../tmp/$(NAME)


CCFLAGS=	-Wall -Wno-deprecated
#Dependecy file creation parameters
DEP_FILE=	$(TMP_DIR)/depend
CDEPFLAGS=	-MM 

LDFLAGS=
ARFLAGS=	-r
INCLUDES=	-I../.. \
		-I$(srcdir)\
		-I../../include\
		-I/usr/include/vtk-5.0\
		-I/usr/include/qt4\
		-I/usr/include/qt4/Qt\
		-I/usr/include/qt4/QtCore\
		-I/usr/include/qt4/QtGui\
		-I/usr/include/qt4/QtOpenGL\
		-I$(VTKIncludeDir)\
		-I/usr/include/vtk\
		-I/usr/include/Qt\
		-I/usr/include/QtGui\
		-I/usr/include/QtCore


MOCOUTPUT=	$(addprefix $(TMP_DIR)/moc_, $(MOCINPUT:.h=.cpp) )

DEBUG_DEFS=	-DDEBUG_LEVEL=10\
		-DDEBUG_ADITIONAL_INFO

DEFS=		$(DEBUG_DEFS)


SRC_FILES=	$(addsuffix .cpp, $(addprefix src/, $(TARGETS)) $(addprefix $(SUBDIR1)/, $(TARGETS1) ) $(addprefix $(SUBDIR2)/, $(TARGETS2) ) $(addprefix $(SUBDIR3)/, $(TARGETS3) ) )
OBJ_FILES=	$(addprefix $(TMP_DIR)/,  $(addprefix moc_, $(MOCINPUT:.h=.o) )  $(addprefix rcc_, $(RCCINPUT:.qrc=.o ) )  $(addsuffix .o, $(TARGETS) $(TARGETS1) $(TARGETS2) $(TARGETS3) ) )


.PHONY: all
all:		tmpdir dep $(OUTPUT)
		

.PHONY: tmpdir
tmpdir:		
		$(MKDIR) $(TMP_DIR) 2>/dev/null || true

.PHONY: build
build:		cleanall all

.PHONY: dep
dep:		
		$(CC) $(INCLUDES) $(CDEPFLAGS) $(SRC_FILES) | \
		sed 's/\(.*\)\.o:/$$(TMP_DIR)\/\1\.o:/' > $(DEP_FILE)

$(TMP_DIR)/moc_%.cpp:	%.h
		$(MOC) $< -o $@

$(TMP_DIR)/moc_%.cpp:	components/%.h
		$(MOC) $< -o $@

$(TMP_DIR)/moc_%.cpp:	utils/%.h
		$(MOC) $< -o $@

$(TMP_DIR)/rcc_%.cpp:	src/%.qrc
		$(RCC) -name $(*F) $< -o $@

$(TMP_DIR)/%.o:	src/%.cpp	
		$(CC) $(CCFLAGS) $(DEFS) $(INCLUDES) -c $< -o $@

$(TMP_DIR)/%.o:	$(SUBDIR1)/%.cpp	
		$(CC) $(CCFLAGS) $(DEFS) $(INCLUDES) -c $< -o $@

$(TMP_DIR)/%.o:	$(SUBDIR2)/%.cpp	
		$(CC) $(CCFLAGS) $(DEFS) $(INCLUDES) -c $< -o $@

$(TMP_DIR)/%.o:	$(SUBDIR3)/%.cpp	
		$(CC) $(CCFLAGS) $(DEFS) $(INCLUDES) -c $< -o $@

$(TMP_DIR)/%.o:	$(TMP_DIR)/%.cpp
		$(CC) $(CCFLAGS) $(DEFS) $(INCLUDES) -c $< -o $@

-include $(DEP_FILE)
		
$(OUTPUT):	$(OBJ_FILES)
		$(AR) $(ARFLAGS) $(OUTPUT) $(OBJ_FILES)

.PHONY: clean
clean:
		$(RM) $(OBJ_FILES) $(MOCOUTPUT) $(addprefix $(TMP_DIR)/rcc_, $(RCCINPUT:.qrc=.cpp ) ) $(DEP_FILE)
		$(RM) $(OUTPUT)
