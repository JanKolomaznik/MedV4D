
srcdir = .
srcTop=..

include $(srcTop)/project/ToolchainSelect.mk
include $(srcTop)/project/configurationSelector.mk
include $(srcTop)/project/pathsDefinitions.mk

########################################
#Name of this project part
NAME=		Imaging$(archPostfix)
########################################

#Name of target
OUTPUTNAME=	lib$(NAME).a
OUTPUTDIR=	$(srcTop)/lib/$(CONF)
OUTPUT=		$(OUTPUTDIR)/$(OUTPUTNAME)
TMP_DIR=	$(srcTop)/tmp/$(CONF)/$(NAME)
SRCDIR=	./src

INCLUDES=	-I$(srcTop)

OBJ_FILES=	$(TMP_DIR)/AbstractDataSet.o \
		$(TMP_DIR)/AbstractFilter.o \
		$(TMP_DIR)/AbstractImageData.o \
		$(TMP_DIR)/AbstractImageFilter.o \
		$(TMP_DIR)/ConnectionInterface.o \
		$(TMP_DIR)/Image.o \
		$(TMP_DIR)/ImageFactory.o \
		$(TMP_DIR)/ModificationManager.o \
		$(TMP_DIR)/PipelineContainer.o \
		$(TMP_DIR)/Pipeline.o \
		$(TMP_DIR)/Ports.o\
		$(TMP_DIR)/DataSetFactory.o

.PHONY: all
all:		tmpdir outDir $(OUTPUT)

.PHONY: build
build:		cleanall all

$(TMP_DIR)/%.o:	$(SRCDIR)/%.cpp
		$(CXX) $(CXXFLAGS) $(DEFS) $(INCLUDES) -c $< -o $@

$(OUTPUT):	$(OBJ_FILES)
		$(AR) $(ARFLAGS) $(OUTPUT) $(OBJ_FILES)

.PHONY: clean
clean:
		$(RM) $(TMP_DIR)/*.o
		$(RM) $(OUTPUT)
