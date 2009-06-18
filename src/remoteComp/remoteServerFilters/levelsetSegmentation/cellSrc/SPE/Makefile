
CELL_TOP=/opt/cell/sdk
srcTop=../../../../..

include $(srcTop)/project/ToolchainSelect.mk
include $(srcTop)/project/pathsDefinitions.mk
include $(srcTop)/project/configurationSelector.mk

########################################
#Name of this project part
NAME=		SPEPart$(archPostfix)
#Names of target sources
TARGETS=	main
########################################

#Name of target
OUTPUTNAME=	lib$(NAME).a
OUTPUTDIR=	$(srcTop)/lib/$(CONF)
OUTPUT=		$(OUTPUTDIR)/$(OUTPUTNAME)
TMP_DIR=	$(srcTop)/tmp/$(CONF)/$(NAME)

INCLUDES +=	-I$(srcTop)\
			-I $(ROOT)/opt/cell/sdk/usr/spu/include
			
LIBS :=
LIB_PATH :=
	
###################### SPE configuration setup ###################		
SPU_LINKER_OPTS = -Wl,-N --gc-sections --print-gc-sections --stack-analysis --local-store=0:256
SPUCXXFLAGS = -W -Wall -Winline -Wno-deprecated -Wstrict-aliasing=2
SPUCXXFLAGS += -fno-exceptions -fno-rtti -fstrict-aliasing 
SPUCXXFLAGS += -DFOR_CELL

ifeq "$(CONF)" "Debug"
SPUCXXFLAGS += -g -DDEBUG_LEVEL=10 -DDEBUG_ADITIONAL_INFO
endif

ifeq "$(CONF)" "Release"
SPUCXXFLAGS += -O3 --param max-unroll-times=1
endif

ifeq "$(CONF)" "PDTProfile"
SPUCXXFLAGS += -g --param max-unroll-times=1 
SPUCXXFLAGS += -Dmain=_pdt_main -Dexit=_pdt_exit -DMFCIO_TRACE -DLIBSYNC_TRACE
SPU_LINKER_OPTS += -Wl,-q -g
INCLUDES += -I$(ROOT)/usr/spu/include/trace
LIB_PATH += -L$(ROOT)/usr/spu/lib/trace
LIBS += -ltrace
endif
ifeq "$(CONF)" "CPCProfile"
SPUCXXFLAGS += -g --param max-unroll-times=1
SPU_LINKER_OPTS += -Wl,-q -g
endif
ifeq "$(CONF)" "SPUTimingToolProfile"
SPUCXXFLAGS += -g -DSPU_TIMING_TOOL_PROFILING
INCLUDES += -I/opt/ibm/systemsim-cell/include/callthru/spu
endif
###################################################################

.PHONY: all
all:		tmpdir SPEbin $(OUTPUT)

.PHONY: build
build:		cleanall all

OBJECT_FILES = 	$(TMP_DIR)/curvatureTermSolver.o\
				$(TMP_DIR)/diffFunc.o\
				$(TMP_DIR)/speedTermSolver.o\
				$(TMP_DIR)/updateCalculatorSPE.o\
				$(TMP_DIR)/applyUpdateCalculator.o\
				$(TMP_DIR)/layerValsPropagator.o\
				$(TMP_DIR)/layerGate.o\
				$(TMP_DIR)/SPEMain.o

$(OUTPUT): SPEbin
	$(EMBEDSPU) SPEMain $(TMP_DIR)/SPEMain $(TMP_DIR)/SPEMain-embed64.o
	$(AR) -qcs $(OUTPUT) $(TMP_DIR)/SPEMain-embed64.o

SPEbin: $(OBJECT_FILES)
	$(SPUCXX) $(SPU_LINKER_OPTS) $(LIB_PATH) $(LIBS) -o $(TMP_DIR)/SPEMain $(OBJECT_FILES)  

## update calc #############
$(TMP_DIR)/curvatureTermSolver.o:
	$(SPUCXX) $(SPUCXXFLAGS) $(INCLUDES) -c -o $(TMP_DIR)/curvatureTermSolver.o updateCalculation/src/curvatureTermSolver.cpp
$(TMP_DIR)/diffFunc.o:
	$(SPUCXX) $(SPUCXXFLAGS) $(INCLUDES) -c -o $(TMP_DIR)/diffFunc.o updateCalculation/src/diffFunc.cpp
$(TMP_DIR)/speedTermSolver.o:
	$(SPUCXX) $(SPUCXXFLAGS) $(INCLUDES) -c -o $(TMP_DIR)/speedTermSolver.o updateCalculation/src/speedTermSolver.cpp
$(TMP_DIR)/updateCalculatorSPE.o:
	$(SPUCXX) $(SPUCXXFLAGS) $(INCLUDES) -c -o $(TMP_DIR)/updateCalculatorSPE.o updateCalculation/src/updateCalculatorSPE.cpp
## apply update calc #############
$(TMP_DIR)/applyUpdateCalculator.o:
	$(SPUCXX) $(SPUCXXFLAGS) $(INCLUDES) -c -o $(TMP_DIR)/applyUpdateCalculator.o applyUpdateCalc/src/applyUpdateCalculator.cpp
$(TMP_DIR)/layerValsPropagator.o:
	$(SPUCXX) $(SPUCXXFLAGS) $(INCLUDES) -c -o $(TMP_DIR)/layerValsPropagator.o applyUpdateCalc/src/layerValsPropagator.cpp
$(TMP_DIR)/layerGate.o:
	$(SPUCXX) $(SPUCXXFLAGS) $(INCLUDES) -c -o $(TMP_DIR)/layerGate.o applyUpdateCalc/src/layerGate.cpp
## main #############
$(TMP_DIR)/SPEMain.o:
	$(SPUCXX) $(SPUCXXFLAGS) $(INCLUDES) -c -o $(TMP_DIR)/SPEMain.o SPEMain.cpp

.PHONY: clean
clean:
		$(RM) $(OBJECT_FILES)
		$(RM) $(TMP_DIR)/SPEMain $(TMP_DIR)/SPEMain-embed64.o
		$(RM) $(OUTPUT)

#MY_SPECIAL_FLAGS = -ffunction-sections -fdata-sections #-g #--param max-unroll-times=1 # needed to keep size of program down
#LDFLAGS_gcc= --gc-sections --print-gc-sections --stack-analysis --local-store=0:256	#-Wl,-q -g
