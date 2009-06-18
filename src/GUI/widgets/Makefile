srcTop = ../..
srcdir = .

include $(srcTop)/project/ToolchainSelect.mk
include $(srcTop)/project/pathsDefinitions.mk
include $(srcTop)/project/configurationSelector.mk

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
		StManagerStudyListComp \
		SimpleSliceViewerTexturePreparer

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
		LoadingThreads.h \
		StudyFilter.h \
		ViewerFactory.h


RCCINPUT=	m4dGUIMainWindow.qrc \
		m4dGUIMainWindow2.qrc \
		m4dGUIStudyManagerWidget.qrc \
		m4dGUIScreenLayoutWidget.qrc \
		m4dGUIToolBarCustomizerWidget.qrc
########################################

#Name of target
OUTPUTNAME=	lib$(NAME).a
OUTPUTDIR=	$(srcTop)/lib/$(CONF)
OUTPUT=		$(OUTPUTDIR)/$(OUTPUTNAME)
TMP_DIR=	$(srcTop)/tmp/$(CONF)/$(NAME)


INCLUDES=	-I$(srcTop)\
			-I$(srcdir)\
			-I$(VTKIncludeDir)\
			$(QTIncludDirs)

MOCOUTPUT=	$(addprefix $(TMP_DIR)/moc_, $(MOCINPUT:.h=.cpp) )


SRC_FILES=	$(addsuffix .cpp, $(addprefix src/, $(TARGETS)) $(addprefix $(SUBDIR1)/, $(TARGETS1) ) $(addprefix $(SUBDIR2)/, $(TARGETS2) ) $(addprefix $(SUBDIR3)/, $(TARGETS3) ) )
OBJ_FILES=	$(addprefix $(TMP_DIR)/,  $(addprefix moc_, $(MOCINPUT:.h=.o) )  $(addprefix rcc_, $(RCCINPUT:.qrc=.o ) )  $(addsuffix .o, $(TARGETS) $(TARGETS1) $(TARGETS2) $(TARGETS3) ) )


.PHONY: all
all:		tmpdir outDir dep $(OUTPUT)

.PHONY: build
build:		cleanall all

.PHONY: dep
dep:
		$(CXX) $(INCLUDES) $(CDEPFLAGS) $(SRC_FILES) | \
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
		$(CXX) $(CXXFLAGS) $(DEFS) $(INCLUDES) -c $< -o $@

$(TMP_DIR)/%.o:	$(SUBDIR1)/%.cpp
		$(CXX) $(CXXFLAGS) $(DEFS) $(INCLUDES) -c $< -o $@

$(TMP_DIR)/%.o:	$(SUBDIR2)/%.cpp
		$(CXX) $(CXXFLAGS) $(DEFS) $(INCLUDES) -c $< -o $@

$(TMP_DIR)/%.o:	$(SUBDIR3)/%.cpp
		$(CXX) $(CXXFLAGS) $(DEFS) $(INCLUDES) -c $< -o $@

$(TMP_DIR)/%.o:	$(TMP_DIR)/%.cpp
		$(CXX) $(CXXFLAGS) $(DEFS) $(INCLUDES) -c $< -o $@

-include $(DEP_FILE)

$(OUTPUT):	$(OBJ_FILES)
		$(AR) $(ARFLAGS) $(OUTPUT) $(OBJ_FILES)

.PHONY: clean
clean:
		$(RM) $(OBJ_FILES) $(MOCOUTPUT) $(addprefix $(TMP_DIR)/rcc_, $(RCCINPUT:.qrc=.cpp ) ) $(DEP_FILE)
		$(RM) $(OUTPUT)
