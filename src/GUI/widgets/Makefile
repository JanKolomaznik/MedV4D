
srcdir = .

CC=g++
AR=ar
RM=rm -f
MKDIR=mkdir
MOC=moc
RCC=rcc

########################################
#Name of this project part
NAME=		m4dWidgets
#Names of target sources
TARGETS=	m4dGUIMainWindow \
		m4dGUIMainViewerDesktopWidget \
		m4dGUIVtkViewerWidget \
		m4dGUIStudyManagerWidget \
		m4dGUIScreenLayoutWidget \
		m4dGUISliceViewerWidget \
		m4dGUISliceViewerWidget2 \
		m4dGUIToolBarCustomizerWidget

SUBDIR=		components
SUBDIR2=	ogl
TARGETS2=	StManagerFilterComp \
		StManagerHProtocolsComp \
		StManagerStudyListComp \
		StudyFilter 
		
TARGETS3=	fonts

MOCINPUT=	m4dGUIAbstractViewerWidget.h \
		m4dGUIMainWindow.h \
		m4dGUIMainViewerDesktopWidget.h \
		m4dGUIScreenLayoutWidget.h \
		m4dGUIStudyManagerWidget.h \
		m4dGUIVtkViewerWidget.h \
		m4dGUISliceViewerWidget.h \
		m4dGUISliceViewerWidget2.h \
		m4dGUIToolBarCustomizerWidget.h \
		StManagerFilterComp.h \
		StManagerHProtocolsComp.h \
		StManagerStudyListComp.h


RCCINPUT=	m4dGUIMainWindow.qrc \
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
		-I/usr/include/qt4/QtOpenGL


MOCOUTPUT=	$(addprefix $(TMP_DIR)/moc_, $(MOCINPUT:.h=.cpp) )

DEBUG_DEFS=	-DDEBUG_LEVEL=10\
		-DDEBUG_ADITIONAL_INFO

DEFS=		$(DEBUG_DEFS)


SRC_FILES=	$(addsuffix .cpp, $(TARGETS) $(addprefix $(SUBDIR)/, $(TARGETS2) ) $(addprefix $(SUBDIR2)/, $(TARGETS3) ) )
OBJ_FILES=	$(addprefix $(TMP_DIR)/,  $(addprefix moc_, $(MOCINPUT:.h=.o) )  $(addprefix rcc_, $(RCCINPUT:.qrc=.o ) )  $(addsuffix .o, $(TARGETS) $(TARGETS2) $(TARGETS3) ) )


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

$(TMP_DIR)/moc_%.cpp:	../../include/GUI/%.h
		$(MOC) $< -o $@

$(TMP_DIR)/rcc_%.cpp:	%.qrc
		$(RCC) -name $(*F) $< -o $@

$(TMP_DIR)/%.o:	%.cpp	
		$(CC) $(CCFLAGS) $(DEFS) $(INCLUDES) -c $< -o $@

$(TMP_DIR)/%.o:	$(SUBDIR)/%.cpp	
		$(CC) $(CCFLAGS) $(DEFS) $(INCLUDES) -c $< -o $@

$(TMP_DIR)/%.o:	$(SUBDIR2)/%.cpp	
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
