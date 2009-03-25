
# used in every makefile to select toolchain that
# will be used in rest of that makefile

##########################################
ifdef COMPILE_FOR_CELL

# subject to change
ITKLibsRoot=/data/cell/ITK/LIBCell
binUtilsPath=/opt/cell/toolchain/bin
ROOT=/opt/cell/sysroot

CXX=$(binUtilsPath)/ppu-g++
SPUComp=$(binUtilsPath)/spu-g++
AR=$(binUtilsPath)/ppu-ar
SPU_AR=$(binUtilsPath)/spu-ar
archPostfix=CELL
CXXDEBUG_OPTIONS= -g
##########################################
else
ITKLibsRoot=/usr/local/lib/InsightToolkit
CXX=g++
AR=ar
ROOT=
archPostfix=
CXXDEBUG_OPTIONS= -ggdb
##########################################
endif
 
