
CXXFLAGS= -Wall -Wno-deprecated
# verbose compiler
#CXXFLAGS += -v
#debug opts
CXXFLAGS += -g -DDEBUG_LEVEL=10 -DDEBUG_ADITIONAL_INFO

# linker flags
LD_PROFILE_OPTS= -Wl,-q
LDFLAGS= $(LD_PROFILE_OPTS)

#Dependecy file creation parameters
DEP_FILE=	$(TMP_DIR)/depend
CDEPFLAGS=	-MM

ARFLAGS=	-r
RM=rm -f
MKDIR=mkdir