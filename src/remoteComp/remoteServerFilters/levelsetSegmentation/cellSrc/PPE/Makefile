
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
OUTPUTDIR=	$(srcTop)/lib
OUTPUT=		$(OUTPUTDIR)/$(OUTPUTNAME)
TMP_DIR=	$(srcTop)/tmp/$(NAME)

INCLUDES=	-I$(srcTop)\
			$(ITKIncludes)\
			-I/opt/cell/sysroot/opt/cell/sdk/usr/include
			
ALTIVEC= -mabi=altivec -maltivec 

.PHONY: all
all:		tmpdir $(OUTPUT)
		

.PHONY: tmpdir
tmpdir:		
		$(MKDIR) $(TMP_DIR) 2>/dev/null || true

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
	$(CXX) $(CXXFLAGS) $(DEFS) $(INCLUDES) $(ALTIVEC) -c -o $(TMP_DIR)/workManager.o src/workManager.cpp

.PHONY: clean
clean:
		$(RM) $(OBJECT_FILES)
		$(RM) $(OUTPUT)
