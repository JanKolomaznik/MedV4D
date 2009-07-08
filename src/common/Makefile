
srcdir = /src
srcTop=..

include $(srcTop)/project/ToolchainSelect.mk
include $(srcTop)/project/configurationSelector.mk
include $(srcTop)/project/pathsDefinitions.mk

########################################
#Name of this project part
NAME=		common$(archPostfix)
########################################

#Name of target
OUTPUTNAME=	lib$(NAME).a
OUTPUTDIR=	$(srcTop)/lib/$(CONF)
OUTPUT=		$(OUTPUTDIR)/$(OUTPUTNAME)
TMP_DIR=	$(srcTop)/tmp/$(CONF)/$(NAME)
SRCDIR := src

INCLUDES=	-I$(srcTop)

OBJ_FILES=	$(TMP_DIR)/Common.o \
		$(TMP_DIR)/Debug.o \
		$(TMP_DIR)/ExceptionBase.o \
		$(TMP_DIR)/Log.o \
		$(TMP_DIR)/Functors.o \
		$(TMP_DIR)/TimeStamp.o \
		$(TMP_DIR)/Thread.o \
		$(TMP_DIR)/perfCounter.o\
		$(TMP_DIR)/fileAccessor.o\
		$(TMP_DIR)/IOStreams.o

.PHONY: all
all:	outDir tmpdir $(OUTPUT)

.PHONY: build
build:	cleanall all

$(TMP_DIR)/%.o:	$(SRCDIR)/%.cpp
		$(CXX) $(CXXFLAGS) $(DEFS) $(INCLUDES) -c $< -o $@

$(OUTPUT):	$(OBJ_FILES)
		$(AR) $(ARFLAGS) $(OUTPUT) $(OBJ_FILES)

.PHONY: clean
clean:
		$(RM) $(TMP_DIR)/*.o $(DEP_FILE)
		$(RM) $(OUTPUT)
