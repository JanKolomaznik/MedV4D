
srcdir = .
srcTop=..

include $(srcTop)/project/ToolchainSelect.mk
include $(srcTop)/project/configurationSelector.mk
include $(srcTop)/project/pathsDefinitions.mk

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
OUTPUTDIR=	$(srcTop)/lib/$(CONF)
OUTPUT=		$(OUTPUTDIR)/$(OUTPUTNAME)
TMP_DIR=	$(srcTop)/tmp/$(CONF)/$(NAME)
SRC_DIR=	./src

INCLUDES=	-I$(srcTop)


SRC_FILES=	$(addprefix $(SRC_DIR)/, $(addsuffix .cpp, $(TARGETS)) )
OBJ_FILES=	$(addprefix $(TMP_DIR)/, $(addsuffix .o, $(TARGETS)) )


.PHONY: all
all:		tmpdir outDir dep $(OUTPUT)

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
