ifndef CONF
CONF = Debug
endif

CXXFLAGS += -Wall -Wno-deprecated -fstrict-aliasing -Wstrict-aliasing=2
# verbose compiler
#CXXFLAGS += -v

####################### CONFs ############################
ifeq "$(CONF)" "Debug"
CXXFLAGS += -g -DDEBUG_LEVEL=10 -DDEBUG_ADITIONAL_INFO
endif

ifeq "$(CONF)" "Release"
CXXFLAGS += -O3
endif

ifeq "$(CONF)" "Profile"
CXXFLAGS += -g -pg -DDEBUG_LEVEL=10 -DDEBUG_ADITIONAL_INFO
endif
##########################################################


# linker flags
#-Wl,-q
LD_PROFILE_OPTS= -pg

ifeq "$(CONF)" "profile"
LDFLAGS= $(LD_PROFILE_OPTS)
endif

#Dependecy file creation parameters
DEP_FILE=	$(TMP_DIR)/depend
CDEPFLAGS=	-MM

ARFLAGS=	-r
RM=rm -f
MKDIR=mkdir