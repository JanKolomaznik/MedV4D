
srcdir = .
srcTop = ../..

include $(srcTop)/project/ToolchainSelect.mk
include $(srcTop)/project/pathsDefinitions.mk
include $(srcTop)/project/configurationSelector.mk

########################################
#Name of this project part
NAME=		levelSetTest$(archPostfix)
#Names of target sources
TARGETS=	main

########################################

#Name of target
OUTPUTNAME=	$(NAME)
OUTPUTDIR=	$(srcTop)/executables/$(CONF)
OUTPUT=		$(OUTPUTDIR)/$(OUTPUTNAME)
TMP_DIR=	$(srcTop)/tmp/$(CONF)/$(NAME)
SRC_DIR=	.

LIBDIRS +=	-L$(srcTop)/lib/$(CONF)\
			-L$(ROOT)/lib64			
				
M4DLIBS=	-lImaging$(archPostfix)\
			-lcommon$(archPostfix)\
			-lPPEStuff$(archPostfix)\
			-lSPEPart$(archPostfix)

LIBS +=		$(M4DLIBS) \
			$(ITKLIBS) \
			-lboost_thread-mt\
			-ldl\
			-lspe2
			
INCLUDES +=	-I$(srcTop) \
			$(ITKIncludes)

SRC_FILES=	$(addprefix $(SRC_DIR)/, $(addsuffix .cpp, $(TARGETS)) )
OBJ_FILES=	$(addprefix $(TMP_DIR)/, $(addsuffix .o, $(TARGETS)) )


.PHONY: all
all:		outDir tmpdir Common Imaging libSPEPart libPPEPart dep $(OUTPUT)
	
Imaging:
	$(MAKE) --directory=$(srcTop)/Imaging all
	
Common:
	$(MAKE) --directory=$(srcTop)/common all
	
libSPEPart:
	$(MAKE) --directory=$(srcTop)/remoteComp/remoteServerFilters/levelsetSegmentation/cellSrc/SPE all
	
libPPEPart:
	$(MAKE) --directory=$(srcTop)/remoteComp/remoteServerFilters/levelsetSegmentation/cellSrc/PPE all
	
##################
	
ImagingCLEAN:
	$(MAKE) --directory=$(srcTop)/Imaging clean
	
CommonCLEAN:
	$(MAKE) --directory=$(srcTop)/common clean
	
libSPEPartCLEAN:
	$(MAKE) --directory=$(srcTop)/remoteComp/remoteServerFilters/levelsetSegmentation/cellSrc/SPE clean
	
libPPEPartCLEAN:
	$(MAKE) --directory=$(srcTop)/remoteComp/remoteServerFilters/levelsetSegmentation/cellSrc/PPE clean

##################

.PHONY: build
build:		cleanall all

.PHONY: dep
dep:
		$(CXX) $(INCLUDES) $(CDEPFLAGS) $(DEFS) $(SRC_FILES) | \
		sed 's/\(.*\)\.o:/$$(TMP_DIR)\/\1\.o:/' > $(DEP_FILE)

$(TMP_DIR)/%.o:	$(SRC_DIR)/%.cpp	
		$(CXX) $(CXXFLAGS) $(DEFS) $(INCLUDES) -c $< -o $@

-include $(DEP_FILE)
		
$(OUTPUT):	$(OBJ_FILES)
		$(CXX) $(CXXFLAGS) $(OBJ_FILES) $(LIBDIRS) $(LIBS) -o $(OUTPUT) 

.PHONY: clean
clean: libSPEPartCLEAN libPPEPartCLEAN #ImagingCLEAN CommonCLEAN
		$(RM) $(TMP_DIR)/*.o $(DEP_FILE)
		$(RM) $(OUTPUT)
