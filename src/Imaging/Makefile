
srcdir = .
projSrcTop=..

ifdef COMPILE_FOR_CELL
binUtilsPath=/opt/cell/toolchain/bin
ROOT=/opt/cell/sysroot

CXX=$(binUtilsPath)/ppu-g++
SPUComp=$(binUtilsPath)/spu-g++
AR=$(binUtilsPath)/ppu-ar
SPU_AR=$(binUtilsPath)/spu-ar
archPostfix=CELL
CXXDEBUG_OPTIONS= -g 
else
CXX=g++
AR=ar
ROOT=
archPostfix=
CXXDEBUG_OPTIONS= -ggdb 
endif

RM=rm -f
MKDIR=mkdir

########################################
#Name of this project part
NAME=		Imaging$(archPostfix)
#Names of target sources
TARGETS=	AbstractDataSet \
		AbstractFilter \
		AbstractImageData \
		AbstractImageFilter \
		ConnectionInterface \
		Image \
		ImageFactory \
		ModificationManager \
		PipelineContainer \
		Pipeline \
		Ports\
		DataSetFactory

#		ImageConnection 
#		ImagePorts 
########################################

#Name of target
OUTPUTNAME=	lib$(NAME).a
OUTPUTDIR=	$(projSrcTop)/lib
OUTPUT=		$(OUTPUTDIR)/$(OUTPUTNAME)
TMP_DIR=	$(projSrcTop)/tmp/$(NAME)
SRC_DIR=	./src

CXXFLAGS= -Wall -Wno-deprecated $(CXXDEBUG_OPTIONS)
#Dependecy file creation parameters
DEP_FILE=	$(TMP_DIR)/depend
CDEPFLAGS=	-MM 

LDFLAGS=
ARFLAGS=	-r
INCLUDES=	-I$(projSrcTop) \
		-I$(projSrcTop)/include\
		-I/usr/local/include/boost-1_38

DEBUG_DEFS=	-DDEBUG_LEVEL=10\
		-DDEBUG_ADITIONAL_INFO

DEFS=		$(DEBUG_DEFS)


SRC_FILES=	$(addprefix $(SRC_DIR)/, $(addsuffix .cpp, $(TARGETS)) )
OBJ_FILES=	$(addprefix $(TMP_DIR)/, $(addsuffix .o, $(TARGETS)) )


.PHONY: all
all:		tmpdir dep $(OUTPUT)
		

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
		$(AR) $(ARFLAGS) $(OUTPUT) $(OBJ_FILES)

.PHONY: clean
clean:
		$(RM) $(TMP_DIR)/*.o $(DEP_FILE)
		$(RM) $(OUTPUT)
