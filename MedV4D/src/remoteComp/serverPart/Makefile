
srcdir = .
srcTop = ../..

include $(srcTop)/project/ToolchainSelect.mk
include $(srcTop)/project/pathsDefinitions.mk
include $(srcTop)/project/configurationSelector.mk

LIBS=		$(ITKLIBS)

########################################
#Name of this project part
NAME=		remoteCompServer$(archPostfix)
########################################

#Name of target
OUTPUTNAME=	lib$(NAME).a
OUTPUTDIR=	$(srcTop)/lib/$(CONF)
OUTPUT=		$(OUTPUTDIR)/$(OUTPUTNAME)
TMP_DIR=	$(srcTop)/tmp/$(CONF)/$(NAME)
SRCDIR=	./src

INCLUDES+= 	-I$(srcTop) \
		$(ITKIncludes)

#Names of target sources
OBJ_FILES=	$(TMP_DIR)/server.o\
		$(TMP_DIR)/remoteFilterFactory.o

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
