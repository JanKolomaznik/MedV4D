
srcdir = .

CXX=g++
AR=ar
RM=rm -f
MKDIR=mkdir

########################################
#Name of this project part
NAME=		backendForDICOM
#Names of target sources
TARGETS=	AbstractService \
		DicomObj \
		FindService \
		DicomAssoc \
		DICOMServiceProvider \
		MoveService \
		StoreService
########################################

#Name of target
OUTPUTNAME=	lib$(NAME).a
OUTPUTDIR=	../lib
OUTPUT=		$(OUTPUTDIR)/$(OUTPUTNAME)
TMP_DIR=	../tmp/$(NAME)


CXXDEBUG_OPTIONS= -ggdb

CXXFLAGS= -Wall -Wno-deprecated $(CXXDEBUG_OPTIONS)
#Dependecy file creation parameters
DEP_FILE=	$(TMP_DIR)/depend
CDEPFLAGS=	-MM 

LDFLAGS=
ARFLAGS=	-r
INCLUDES=	-I$(srcdir)\
		-I../include


DEBUG_DEFS=	-DDEBUG_LEVEL=10\
		-DDEBUG_ADITIONAL_INFO
		

DEFS=		-DHAVE_CONFIG_H\
		-DHAVE_SSTREAM\
		$(DEBUG_DEFS)


SRC_FILES=	$(addsuffix .cpp, $(TARGETS))
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

$(TMP_DIR)/%.o:	%.cpp	
		$(CXX) $(CXXFLAGS) $(DEFS) $(INCLUDES) -c $< -o $@

-include $(DEP_FILE)
		
$(OUTPUT):	$(OBJ_FILES)
		$(AR) $(ARFLAGS) $(OUTPUT) $(OBJ_FILES)

.PHONY: clean
clean:
		$(RM) $(TMP_DIR)/*.o $(DEP_FILE)
		$(RM) $(OUTPUT)
