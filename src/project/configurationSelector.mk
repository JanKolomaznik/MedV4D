ifndef CONF
CONF = Debug
endif

CXXFLAGS += -Wall -Wno-deprecated -fstrict-aliasing -Wstrict-aliasing=2
# verbose compiler
#CXXFLAGS += -v

# linker flags
#-Wl,-q
LDFLAGS=

####################### CONFs ############################
ifeq "$(CONF)" "Debug"
CXXFLAGS += -g -DDEBUG_LEVEL=10 -DDEBUG_ADITIONAL_INFO
endif

ifeq "$(CONF)" "Release"
CXXFLAGS += -O3
endif

ifeq "$(CONF)" "PDTProfile"
#----------------- PDT profiling stuff ------------------
INCLUDES += -I$(ROOT)/usr/include/trace
LIBS += -L$(ROOT)/usr/lib64/trace
CXXFLAGS += -g -DDEBUG_LEVEL=10 -DDEBUG_ADITIONAL_INFO
LDFLAGS += -g -Wl,-q
#--------------------------------------------------------
endif

ifeq "$(CONF)" "CPCProfile"
#----------------- CPC profiling stuff ------------------
CXXFLAGS += -g -DDEBUG_LEVEL=10 -DDEBUG_ADITIONAL_INFO
LDFLAGS += -g -Wl,-q
#--------------------------------------------------------
endif

ifeq "$(CONF)" "SPUTimingToolProfile"
#----------- SPUTimingTool profiling stuff --------------
# the same as Debug. Profiling is only for SPE
# So appropriate defs are in SPE makefile
CXXFLAGS += -g -DDEBUG_LEVEL=10 -DDEBUG_ADITIONAL_INFO -DSPU_TIMING_TOOL_PROFILING
#--------------------------------------------------------
endif
##########################################################

#Dependecy file creation parameters
DEP_FILE=	$(TMP_DIR)/depend
CDEPFLAGS=	-MM