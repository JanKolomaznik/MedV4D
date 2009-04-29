
# used in every makefile to select toolchain that
# will be used in rest of that makefile

##########################################
ifdef COMPILE_FOR_CELL

binUtilsPath=/opt/cell/toolchain/bin
ROOT=/opt/cell/sysroot
PLATFORM_DEFS= -DFOR_CELL

CXX=$(binUtilsPath)/ppu-g++
SPU_LD_PROFILE_OPTS= -Wl,-q -g
SPUCXX=$(binUtilsPath)/spu-g++ -g --param max-unroll-times=1
AR=$(binUtilsPath)/ppu-ar
SPU_AR=$(binUtilsPath)/spu-ar
archPostfix=CELL
##########################################
else ifdef COMPILE_ON_CELL
# not tested
CXX=ppu-g++
AR=ppu-ar
ROOT=
archPostfix=CELL
PLATFORM_DEFS= -DFOR_CELL

##########################################
else

CXX=g++
AR=ar
ROOT=
archPostfix=

PLATFORM_DEFS= -DFOR_PC

CXXDEBUG_OPTIONS= -ggdb
##########################################
endif

RM=rm -f
MKDIR=mkdir
MOC=moc
RCC=rcc
ARFLAGS=	-r