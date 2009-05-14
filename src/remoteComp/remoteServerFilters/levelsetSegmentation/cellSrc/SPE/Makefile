
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
OUTPUTDIR=	$(srcTop)/lib
OUTPUT=		$(OUTPUTDIR)/SPU/$(OUTPUTNAME)
TMP_DIR=	$(srcTop)/tmp/$(NAME)

INCLUDES=	-I$(srcTop)\
			-I /opt/cell/sysroot/opt/cell/sdk/usr/spu/include
			
SPU_LINKER_OPTS = -Wl,-N --gc-sections --print-gc-sections --stack-analysis --local-store=0:256

CXXFLAGS += -W -Winline -fno-exceptions -fno-rtti
#SPU_COMPILER_OPTS += --param max-unroll-times=1

.PHONY: all
all:		tmpdir SPEbin $(OUTPUT)


.PHONY: tmpdir
tmpdir:
		$(MKDIR) $(TMP_DIR) 2>/dev/null || true

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
	$(SPUCXX) $(SPU_LINKER_OPTS) -o $(TMP_DIR)/SPEMain $(OBJECT_FILES)  

## update calc #############
$(TMP_DIR)/curvatureTermSolver.o:
	$(SPUCXX) $(CXXFLAGS) $(INCLUDES) -c -o $(TMP_DIR)/curvatureTermSolver.o updateCalculation/src/curvatureTermSolver.cpp
$(TMP_DIR)/diffFunc.o:
	$(SPUCXX) $(CXXFLAGS) $(INCLUDES) -c -o $(TMP_DIR)/diffFunc.o updateCalculation/src/diffFunc.cpp
$(TMP_DIR)/speedTermSolver.o:
	$(SPUCXX) $(CXXFLAGS) $(INCLUDES) -c -o $(TMP_DIR)/speedTermSolver.o updateCalculation/src/speedTermSolver.cpp
$(TMP_DIR)/updateCalculatorSPE.o:
	$(SPUCXX) $(CXXFLAGS) $(INCLUDES) -c -o $(TMP_DIR)/updateCalculatorSPE.o updateCalculation/src/updateCalculatorSPE.cpp
## apply update calc #############
$(TMP_DIR)/applyUpdateCalculator.o:
	$(SPUCXX) $(CXXFLAGS) $(INCLUDES) -c -o $(TMP_DIR)/applyUpdateCalculator.o applyUpdateCalc/src/applyUpdateCalculator.cpp
$(TMP_DIR)/layerValsPropagator.o:
	$(SPUCXX) $(CXXFLAGS) $(INCLUDES) -c -o $(TMP_DIR)/layerValsPropagator.o applyUpdateCalc/src/layerValsPropagator.cpp
$(TMP_DIR)/layerGate.o:
	$(SPUCXX) $(CXXFLAGS) $(INCLUDES) -c -o $(TMP_DIR)/layerGate.o applyUpdateCalc/src/layerGate.cpp
## main #############
$(TMP_DIR)/SPEMain.o:
	$(SPUCXX) $(CXXFLAGS) $(INCLUDES) -c -o $(TMP_DIR)/SPEMain.o SPEMain.cpp

.PHONY: clean
clean:
		$(RM) $(OBJECT_FILES)
		$(RM) $(TMP_DIR)/SPEMain $(TMP_DIR)/SPEMain-embed64.o
		$(RM) $(OUTPUT)

#MY_SPECIAL_FLAGS = -ffunction-sections -fdata-sections #-g #--param max-unroll-times=1 # needed to keep size of program down
#LDFLAGS_gcc= --gc-sections --print-gc-sections --stack-analysis --local-store=0:256	#-Wl,-q -g
