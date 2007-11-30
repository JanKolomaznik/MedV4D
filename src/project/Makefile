##############################################################
#   Makefile for MedV4D (pilot)
#   $Rev$
#   $Date$

###########################################
#   COMPILER:
# You can use any ANSII C++ compiler

#CC = g++
CC = gcc
#CC = CC

#LINKER = cc
LINKER = ${CC}

###########################################
#   compiler OPTIONS:

# For debugging, change "-O" to "-g"
DBX = -O3
#DBX = -O3 -mpentium
#DBX = -O3 -mcpu=athlon64 -m3dnow
#DBX = -O3 -mcpu=athlon -m3dnow
#DBX = -O3 -mtune=prescott -msse3

# Any other compiler or linker options?
CFLAGS =
LDFLAGS =

# Add -DSYMBOL to define preprocessor symbol:
DEFINES =
#DEFINES = -DUSE_64_BITS
#DEFINES = -DALT

###########################################
#   LIBRARIES:

#LIBS = -lm
LIBS = -lm -lstdc++ -lpthread -L../lib

###########################################
#   INCLUDES:

# To change where the include files come from, change INCLUDE.
INCLUDE = -I../include

##############################################################
#  Usually you don't have to edit anything below this line..

# rm command for clean target:
RM = rm -f

# object files to build "bench":
OBJS = ../xxxx/blablabla.o

CMDS = medv4d

all:   ${CMDS}

.cpp.o:
	${CC} $(DEFINES) $(CFLAGS) $(INCLUDE) $(DBX) -c $<

medv4d:	${OBJS}
	${LINKER} ${LDFLAGS} ${DBX} ${OBJS} ${LIBS} -o ../bin/medv4d
	strip ../bin/medv4d

clean:
	${RM} ../bin/*.o ../xxxx/*% ../xxxx/*~ ../bin/core
