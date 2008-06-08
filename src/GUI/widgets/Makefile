
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
TARGETS=	m4dGUIVtkRenderWindowWidget \
		m4dGUIStudyManagerWidget \
		m4dGUIScreenLayoutWidget

SUBDIR=		components
TARGETS2=	StManagerFilterComp \
		StManagerHProtocolsComp \
		StManagerStudyListComp

MOCINPUT=	m4dGUIStudyManagerWidget.h \
		m4dGUIVtkRenderWindowWidget.h \
		m4dGUIScreenLayoutWidget.h \
		StManagerFilterComp.h \
		StManagerHProtocolsComp.h \
		StManagerStudyListComp.h

RCCINPUT=	m4dGUIStudyManagerWidget.qrc \
		m4dGUIScreenLayoutWidget.qrc
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
INCLUDES=	-I$(srcdir)\
		-I../../include\
		-I/usr/include/vtk-5.0\
		-I/usr/include/qt4\
		-I/usr/include/qt4/Qt\
		-I/usr/include/qt4/QtCore\
		-I/usr/include/qt4/QtGui


MOCOUTPUT=	$(addprefix $(TMP_DIR)/moc_, $(MOCINPUT:.h=.cpp) )

DEBUG_DEFS=	-DDEBUG_LEVEL=10\
		-DDEBUG_ADITIONAL_INFO

DEFS=		$(DEBUG_DEFS)


SRC_FILES=	$(addsuffix .cpp, $(TARGETS) $(addprefix $(SUBDIR)/, $(TARGETS2) ) ) 
OBJ_FILES=	$(addprefix $(TMP_DIR)/,  $(addprefix moc_, $(MOCINPUT:.h=.o) )  $(addprefix rcc_, $(RCCINPUT:.qrc=.o ) )  $(addsuffix .o, $(TARGETS) $(TARGETS2) ) )


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

$(TMP_DIR)/rcc_%.cpp:	$(RCCINPUT)
		$(RCC) -name $(*F) $< -o $@

$(TMP_DIR)/%.o:	%.cpp	
		$(CC) $(CCFLAGS) $(DEFS) $(INCLUDES) -c $< -o $@

$(TMP_DIR)/%.o:	$(SUBDIR)/%.cpp	
		$(CC) $(CCFLAGS) $(DEFS) $(INCLUDES) -c $< -o $@

$(TMP_DIR)/%.o:	$(TMP_DIR)/%.cpp
		$(CC) $(CCFLAGS) $(DEFS) $(INCLUDES) -c $< -o $@

-include $(DEP_FILE)
		
$(OUTPUT):	$(OBJ_FILES)
		$(AR) $(ARFLAGS) $(OUTPUT) $(OBJ_FILES)

.PHONY: clean
clean:
		$(RM) $(OBJ_FILES) $(MOCOUTPUT) $(addprefix $(TMP_DIR)/rcc_, $(RCCINPUT:.qrc=.cpp ) ) $(DEP_FILE)

.PHONY: cleanall
cleanall:	clean
		$(RM) $(OUTPUT)
