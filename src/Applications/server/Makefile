
srcdir = .
srcTop = ../..

include $(srcTop)/project/ToolchainSelect.mk
include $(srcTop)/project/configurationSelector.mk
include $(srcTop)/project/pathsDefinitions.mk

########################################
#Name of this project part
NAME=		remoteCompServer$(archPostfix)
#Names of target sources
TARGETS=	main

########################################

#Name of target
OUTPUTNAME=	$(NAME)
OUTPUTDIR=	$(srcTop)/executables/$(CONF)
OUTPUT=		$(OUTPUTDIR)/$(OUTPUTNAME)
TMP_DIR=	$(srcTop)/tmp/$(CONF)/$(NAME)
SRC_DIR=	.

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

SRC_FILES=	$(addprefix $(SRC_DIR)/, $(addsuffix .cpp, $(TARGETS)) )
OBJ_FILES=	$(addprefix $(TMP_DIR)/, $(addsuffix .o, $(TARGETS)) )

printInfo:
	@echo "*************************"
	@echo "*"
	@echo "* configuration: $(CONF)"
	@echo "* architecture: $(ARCH)"
	@echo "*"
	@echo "*************************"


.PHONY: all
all:		printInfo tmpdir Common Imaging commonRCPart serverRCPart libSPUPart PPEStuf dep $(OUTPUT)

commonRCPart:
	$(MAKE) --directory=$(srcTop)/remoteComp/commonPart all
	
serverRCPart:
	$(MAKE) --directory=$(srcTop)/remoteComp/serverPart all
	
Imaging:
	$(MAKE) --directory=$(srcTop)/Imaging all
	
Common:
	$(MAKE) --directory=$(srcTop)/common all
	
libSPUPart:
	$(MAKE) --directory=$(srcTop)/remoteComp/remoteServerFilters/levelsetSegmentation/cellSrc/SPE all
	
PPEStuf:
	$(MAKE) --directory=$(srcTop)/remoteComp/remoteServerFilters/levelsetSegmentation/cellSrc/PPE all
	
##################

commonRCPartCLEAN:
	$(MAKE) --directory=$(srcTop)/remoteComp/commonPart clean
	
serverRCPartCLEAN:
	$(MAKE) --directory=$(srcTop)/remoteComp/serverPart clean
	
ImagingCLEAN:
	$(MAKE) --directory=$(srcTop)/Imaging clean
	
CommonCLEAN:
	$(MAKE) --directory=$(srcTop)/common clean
	
libSPUPartCLEAN:
	$(MAKE) --directory=$(srcTop)/remoteComp/remoteServerFilters/levelsetSegmentation/cellSrc/SPE clean
	
PPEStufCLEAN:
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
clean: libSPUPartCLEAN PPEStufCLEAN serverRCPartCLEAN ImagingCLEAN CommonCLEAN
		$(RM) $(TMP_DIR)/*.o $(DEP_FILE)
		$(RM) $(OUTPUT)
