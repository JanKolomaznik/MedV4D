srcTop = ..
srcdir = .

include $(srcTop)/project/ToolchainSelect.mk
include $(srcTop)/project/pathsDefinitions.mk
include $(srcTop)/project/configurationSelector.mk

########################################
#Name of this project part
NAME=		backendForDICOM
########################################

#Name of target
OUTPUTNAME=	lib$(NAME).a
OUTPUTDIR=	$(srcTop)/lib/$(CONF)
OUTPUT=		$(OUTPUTDIR)/$(OUTPUTNAME)
TMP_DIR=	$(srcTop)/tmp/$(CONF)/$(NAME)
SRCDIR=	./src

INCLUDES +=	-I$(srcTop) \
		-I$(srcdir)\
		-I$(DCMTK_INCLUDE_PATH)

CXXFLAGS +=	-DHAVE_CONFIG_H\
		-DHAVE_SSTREAM\

OBJ_FILES=	$(TMP_DIR)/AbstractService.o\
		$(TMP_DIR)/DicomAssoc.o \
		$(TMP_DIR)/DicomObj.o \
		$(TMP_DIR)/DICOMServiceProvider.o \
		$(TMP_DIR)/FindService.o \
		$(TMP_DIR)/LocalService.o \
		$(TMP_DIR)/MoveService.o \
		$(TMP_DIR)/StoreService.o

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
