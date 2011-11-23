
srcdir = .
srcTop = ../..

include $(srcTop)/project/ToolchainSelect.mk
include $(srcTop)/project/configurationSelector.mk
include $(srcTop)/project/pathsDefinitions.mk

########################################
#Name of this project part
NAME=		MedvedServer$(archPostfix)
########################################

#Name of target
OUTPUTNAME=	$(NAME)
OUTPUTDIR=	$(srcTop)/executables/$(CONF)
OUTPUT=		$(OUTPUTDIR)/$(OUTPUTNAME)
TMP_DIR=	$(srcTop)/tmp/$(CONF)/$(NAME)
SRCDIR=	.

M4DLIBS=	-lremoteCompServer$(archPostfix)\
		-lremoteCompCommons$(archPostfix) \
		-lImaging$(archPostfix)\
		-lcommon$(archPostfix)\
		-lPPEStuff$(archPostfix) \
		-lSPEPart$(archPostfix)

LIBS=		$(M4DLIBS) \
		$(ITKLIBS) \
		-lboost_thread-mt\
		-ldl\
		-lspe2

LIBDIRS=	-L$(srcTop)/lib/$(CONF)\
		-L$(ROOT)/lib64

INCLUDES+=	-I$(srcTop) \
		$(ITKIncludes)

#Names of target sources
OBJ_FILES=	$(TMP_DIR)/main.o

.PHONY: all
all:		tmpdir libSPUPart PPEStuf $(OUTPUT)

.PHONY: build
build:		cleanall all

libSPUPart:
	$(MAKE) --directory=$(srcTop)/remoteComp/remoteServerFilters/levelsetSegmentation/cellSrc/SPE all

PPEStuf:
	$(MAKE) --directory=$(srcTop)/remoteComp/remoteServerFilters/levelsetSegmentation/cellSrc/PPE all

$(TMP_DIR)/%.o:	$(SRCDIR)/%.cpp
		$(CXX) $(CXXFLAGS) $(DEFS) $(INCLUDES) -c $< -o $@

$(OUTPUT):	$(OBJ_FILES)
		$(CXX) $(CXXFLAGS) $(OBJ_FILES) $(LIBDIRS) $(LIBS) -o $(OUTPUT)

.PHONY: clean
clean:
	$(RM) $(TMP_DIR)/*.o
	$(RM) $(OUTPUT)
	$(MAKE) --directory=$(srcTop)/remoteComp/remoteServerFilters/levelsetSegmentation/cellSrc/SPE clean
	$(MAKE) --directory=$(srcTop)/remoteComp/remoteServerFilters/levelsetSegmentation/cellSrc/PPE clean