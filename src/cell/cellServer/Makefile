
srcdir = .

CXX=g++
AR=ar
RM=rm -f
MKDIR=mkdir

########################################
#Name of this project part
NAME=		cellServer
#Names of target sources
TARGETS=	server \
		main \
		serverJob

M4DLIBS=	../../lib/libcellCommon.a \
		../../lib/libImaging.a\
		../../lib/libbackendForDICOM.a\
		../../lib/libcommon.a

LIBS=		$(M4DLIBS) \
		-lboost_thread \
		-lboost_filesystem

########################################

#Name of target
OUTPUTNAME=	$(NAME)
OUTPUTDIR=	../../executables
OUTPUT=		$(OUTPUTDIR)/$(OUTPUTNAME)
TMP_DIR=	../../tmp/$(NAME)
SRC_DIR=	.

CXXDEBUG_OPTIONS= -ggdb 

CXXFLAGS= -Wall -Wno-deprecated $(CXXDEBUG_OPTIONS)
#Dependecy file creation parameters
DEP_FILE=	$(TMP_DIR)/depend
CDEPFLAGS=	-MM 

LDFLAGS=
ARFLAGS=	-r
LIBDIRS=	-L../../lib 

INCLUDES=	-I../.. \
		-I../../include



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
		$(CXX) $(CXXFLAGS) $(OBJ_FILES) $(LIBDIRS) $(LIBS) -o $(OUTPUT) 

.PHONY: clean
clean:
		$(RM) $(TMP_DIR)/*.o $(DEP_FILE)
		$(RM) $(OUTPUT)
