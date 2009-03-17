
srcdir = .
srcTop = ../..

include $(srcTop)/project/ToolchainSelect.mk

RM=rm -f
MKDIR=mkdir

ITKIncludeRoot=/usr/local/include/InsightToolkit

ITKIncludes=	-I$(ITKIncludeRoot)\
				-I$(ITKIncludeRoot)/Common\
				-I$(ITKIncludeRoot)/Algorithms\
				-I$(ITKIncludeRoot)/BasicFilters\
				-I$(ITKIncludeRoot)/IO\
				-I$(ITKIncludeRoot)/Utilities\
				-I$(ITKIncludeRoot)/Utilities/vxl/vcl\
				-I$(ITKIncludeRoot)/Utilities/vxl/core\
				-I$(ITKIncludeRoot)/gdcm/src
				
ITKLIBS= 	$(ITKLibsRoot)/libITKNumerics.a\
			$(ITKLibsRoot)/libITKNrrdIO.a\
			$(ITKLibsRoot)/libITKSpatialObject.a\
			$(ITKLibsRoot)/libITKCommon.a\
			$(ITKLibsRoot)/libitkvnl_inst.a\
			$(ITKLibsRoot)/libitkvnl_algo.a\
			$(ITKLibsRoot)/libitkvnl.a\
			$(ITKLibsRoot)/libitkv3p_netlib.a\
			$(ITKLibsRoot)/libitkvcl.a\
			$(ITKLibsRoot)/libitksys.a\
			$(ITKLibsRoot)/libITKMetaIO.a\
			$(ITKLibsRoot)/libITKEXPAT.a\
			$(ITKLibsRoot)/libITKniftiio.a\
			$(ITKLibsRoot)/libITKznz.a\
				
M4DLIBS=	$(srcTop)/lib/libremoteCompCommons$(archPostfix).a \
		$(srcTop)/lib/libImaging$(archPostfix).a\
		$(srcTop)/lib/libcommon$(archPostfix).a\
		$(srcTop)/lib/remoteCompServer$(archPostfix).a

LIBS=		$(M4DLIBS) \
			$(ITKLIBS) \
		-lboost_thread \
		-lboost_filesystem\
		-ldl

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

CXXFLAGS= -Wall -Wno-deprecated $(CXXDEBUG_OPTIONS)
#Dependecy file creation parameters
DEP_FILE=	$(TMP_DIR)/depend
CDEPFLAGS=	-MM 

LDFLAGS=
ARFLAGS=	-r
LIBDIRS=	-L$(srcTop)/lib

INCLUDES=	-I$(srcTop) \
			-I$(srcTop)/include\
			$(ITKIncludes)\
			-I/usr/local/include/boost-1_38

DEBUG_DEFS=	-DDEBUG_LEVEL=10\
			-DDEBUG_ADITIONAL_INFO

DEFS=		$(DEBUG_DEFS)


SRC_FILES=	$(addprefix $(SRC_DIR)/, $(addsuffix .cpp, $(TARGETS)) )
OBJ_FILES=	$(addprefix $(TMP_DIR)/, $(addsuffix .o, $(TARGETS)) )


.PHONY: all
all:		tmpdir commonPart serverPart dep $(OUTPUT)
		
commonPart:
	$(MAKE) --directory=$(srcTop)/remoteComp/commonPart
	
serverPart:
	$(MAKE) --directory=$(srcTop)/remoteComp/serverPart

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
clean:
		$(RM) $(TMP_DIR)/*.o $(DEP_FILE)
		$(RM) $(OUTPUT)
