
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
LD_PROFILE_OPTS= -Wl,-q
CXXDEBUG_OPTIONS= -g
##########################################
else ifdef COMPILE_ON_CELL

CXX=ppu-g++
AR=ppu-ar
ROOT=
archPostfix=CELL
PLATFORM_DEFS= -DFOR_CELL

PROFILE_OPTIONS=
CXXDEBUG_OPTIONS= -g
##########################################
else

CXX=g++
AR=ar
ROOT=
archPostfix=

PLATFORM_DEFS= -DFOR_PC
PROFILE_OPTIONS=
CXXDEBUG_OPTIONS= -ggdb $(PROFILE_OPTIONS)
##########################################
endif
 
