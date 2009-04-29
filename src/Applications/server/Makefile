
srcdir = .
srcTop = ../..

include $(srcTop)/project/ToolchainSelect.mk
include $(srcTop)/project/pathsDefinitions.mk
include $(srcTop)/project/configurationSelector.mk

RM=rm -f
MKDIR=mkdir
				
M4DLIBS=	$(srcTop)/lib/libremoteCompServer$(archPostfix).a\
		$(srcTop)/lib/libremoteCompCommons$(archPostfix).a \
		$(srcTop)/lib/libImaging$(archPostfix).a\
		$(srcTop)/lib/libcommon$(archPostfix).a\
		$(srcTop)/lib/libPPEStuff$(archPostfix).a \
		$(srcTop)/lib/SPU/libSPEPart$(archPostfix).a

LIBS=		$(M4DLIBS) \
		$(ITKLIBS) \
		-lboost_thread\
		-lboost_filesystem\
		-ldl\
		-lspe2

########################################
#Name of this project part
NAME=		remoteCompServer$(archPostfix)
#Names of target sources
TARGETS=	main

########################################

#Name of target
OUTPUTNAME=	$(NAME)
OUTPUTDIR=	$(srcTop)/executables
OUTPUT=		$(OUTPUTDIR)/$(OUTPUTNAME)
TMP_DIR=	$(srcTop)/tmp/$(NAME)
SRC_DIR=	.

LIBDIRS=	-L$(srcTop)/lib\
			-L$(ROOT)/lib64
			
INCLUDES=	-I$(srcTop) \
		$(ITKIncludes)

SRC_FILES=	$(addprefix $(SRC_DIR)/, $(addsuffix .cpp, $(TARGETS)) )
OBJ_FILES=	$(addprefix $(TMP_DIR)/, $(addsuffix .o, $(TARGETS)) )


.PHONY: all
all:		tmpdir Common Imaging commonRCPart serverRCPart libSPUPart PPEStuf dep $(OUTPUT)

commonRCPart:
	$(MAKE) --directory=$(srcTop)/remoteComp/commonPart
	
serverRCPart:
	$(MAKE) --directory=$(srcTop)/remoteComp/serverPart
	
Imaging:
	$(MAKE) --directory=$(srcTop)/Imaging
	
Common:
	$(MAKE) --directory=$(srcTop)/common
	
libSPUPart:
	$(MAKE) --directory=$(srcTop)/remoteComp/remoteServerFilters/levelsetSegmentation/cellSrc/SPE
	
PPEStuf:
	$(MAKE) --directory=$(srcTop)/remoteComp/remoteServerFilters/levelsetSegmentation/cellSrc/PPE
	
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

.PHONY: tmpdir
tmpdir:		
		$(MKDIR) $(TMP_DIR) 2>/dev/null || true

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
clean: libSPUPartCLEAN PPEStufCLEAN # serverRCPartCLEAN ImagingCLEAN CommonCLEAN
		$(RM) $(TMP_DIR)/*.o $(DEP_FILE)
		$(RM) $(OUTPUT)
