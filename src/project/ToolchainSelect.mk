
# used in every makefile to select toolchain that
# will be used in rest of that makefile

ifndef ARCH
ARCH = PC
endif

##########################################
ifeq "$(ARCH)" "CellCross"

binUtilsPath=/opt/cell/toolchain/bin
ROOT=/opt/cell/sysroot
CXXFLAGS += -DFOR_CELL

CXX=$(binUtilsPath)/ppu-g++
SPU_LD_PROFILE_OPTS= -Wl,-q -g
SPUCXX=$(binUtilsPath)/spu-g++ -g --param max-unroll-times=1
AR=$(binUtilsPath)/ppu-ar
SPU_AR=$(binUtilsPath)/spu-ar
archPostfix=CELL
endif
##########################################
#used in porting phase
ifeq "$(ARCH)" "CellPCTest"
binUtilsPath=/opt/cell/toolchain/bin
ROOT=/opt/cell/sysroot
CXX=$(binUtilsPath)/ppu-g++
AR=$(binUtilsPath)/ppu-ar
archPostfix=CELL
CXXFLAGS += -DFOR_PC
endif
##########################################
ifeq "$(ARCH)" "CellNative"
# not tested
CXX=ppu-g++
AR=ppu-ar
ROOT=
archPostfix=CELL
CXXFLAGS += -DFOR_CELL
endif
##########################################
ifeq "$(ARCH)" "PC"

CXX=g++
AR=ar
ROOT=
archPostfix=

CXXFLAGS += -DFOR_PC

CXXDEBUG_OPTIONS= -ggdb
endif
##########################################

RM=rm -f
MKDIR=mkdir
MOC=moc
RCC=rcc
ARFLAGS=	-r