
srcTop=../../../../..

include $(srcTop)/project/ToolchainSelect.mk
include $(srcTop)/project/pathsDefinitions.mk
include $(srcTop)/project/configurationSelector.mk

########################################
#Name of this project part
NAME=		PPEStuff$(archPostfix)
########################################

#Name of target
OUTPUTNAME=	lib$(NAME).a
OUTPUTDIR=	$(srcTop)/lib/$(CONF)
OUTPUT=		$(OUTPUTDIR)/$(OUTPUTNAME)
TMP_DIR=	$(srcTop)/tmp/$(CONF)/$(NAME)

INCLUDES=	-I$(srcTop)

.PHONY: all
all:		outDir tmpdir $(OUTPUT)

.PHONY: build
build:		cleanall all

OBJECT_FILES = 	$(TMP_DIR)/SPEManager.o\
				$(TMP_DIR)/SPURequestsDispatcher.o\
				$(TMP_DIR)/workManager.o
		
$(OUTPUT): $(OBJECT_FILES)
		$(AR) $(ARFLAGS) $(OUTPUT) $(OBJECT_FILES)

$(TMP_DIR)/SPEManager.o:
	$(CXX) $(CXXFLAGS) $(DEFS) $(INCLUDES) -c -o $(TMP_DIR)/SPEManager.o src/SPEManager.cpp
$(TMP_DIR)/SPURequestsDispatcher.o:
	$(CXX) $(CXXFLAGS) $(DEFS) $(INCLUDES) -c -o $(TMP_DIR)/SPURequestsDispatcher.o src/SPURequestsDispatcher.cpp
$(TMP_DIR)/workManager.o:
	$(CXX) $(CXXFLAGS) $(DEFS) $(INCLUDES) -c -o $(TMP_DIR)/workManager.o src/workManager.cpp

.PHONY: clean
clean:
		$(RM) $(OBJECT_FILES)
		$(RM) $(OUTPUT)
