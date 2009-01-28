
srcdir = .
srcroot = ../../..

ifdef COMPILE_FOR_CELL
############################### CellBE
binUtilsPath=/opt/cell/toolchain/bin
ROOT=/opt/cell/sysroot

CXX=$(binUtilsPath)/ppu-g++
SPUComp=$(binUtilsPath)/spu-g++
AR=$(binUtilsPath)/ppu-ar
SPU_AR=$(binUtilsPath)/spu-ar
archPostfix=CELL

INCLUDES=	-I$(ROOT)/usr/include\
			-I$(binUtilsPath)/lib/gcc/ppu/4.1.1/include\
			-I$(binUtilsPath)/lib/gcc/ppu/4.1.1/include/c++\
			-I$(srcdir)\
			-I$(srcroot)/include\
			-I$(srcroot)\

LIBDIRS=	-L$(ROOT)/lib64

M4DLIBS=	$(srcroot)/lib/libImaging$(archPostfix).a\
		$(srcroot)/lib/libcommon$(archPostfix).a

LIBS=	$(M4DLIBS)\
		$(srcroot)/lib/libboost_threadCELL.a \
		$(srcroot)/lib/libboost_filesystemCELL.a\
		-lpthread
		#$(ROOT)/usr/lib64/libpthread_nonshared.a
		
else
##################### NORMAL PC
CXX=g++
AR=ar
ROOT=
archPostfix=
LIBDIRS=	-L$(srcroot)/lib\
			-L/usr/local/lib/

INCLUDES=	-I$(srcdir)\
			-I$(srcroot)/include\
			-I$(srcroot)\
			-I/usr/local/include/boost-1_38
			
M4DLIBS=	$(srcroot)/lib/libImaging.a\
		$(srcroot)/lib/libcommon.a

LIBS=	$(M4DLIBS)\
		-lboost_thread\
		-lboost_filesystem
endif

RM=rm -f
MKDIR=mkdir

########################################
#Name of this project part
NAME=		cellPok1$(archPostfix)
#Names of target sources
TARGETS=	main
########################################

#Name of target
OUTPUTNAME=	$(NAME)
OUTPUTDIR=	$(srcroot)/executables
OUTPUT=		$(OUTPUTDIR)/$(OUTPUTNAME)
TMP_DIR=	$(srcroot)/tmp/$(NAME)

CXXDEBUG_OPTIONS= -ggdb 

CXXFLAGS= -Wall -Wno-deprecated $(CXXDEBUG_OPTIONS)
#Dependecy file creation parameters
DEP_FILE=	$(TMP_DIR)/depend
CDEPFLAGS=	-MM 

LDFLAGS=
ARFLAGS=	-r

DEBUG_DEFS=	-DDEBUG_LEVEL=10\
		-DDEBUG_ADITIONAL_INFO

DEFS=		$(DEBUG_DEFS)

SRC_FILES=	$(addsuffix .cpp, $(TARGETS) ) 
OBJ_FILES=	$(addprefix $(TMP_DIR)/,   $(addprefix rcc_, $(RCCINPUT:.qrc=.o ) ) $(addprefix moc_, $(MOCINPUT:.h=.o) ) $(addsuffix .o, $(TARGETS) ) )


.PHONY: all
all:		tmpdir dep $(OUTPUT)
		

.PHONY: tmpdir
tmpdir:		
		$(MKDIR) $(TMP_DIR) 2>/dev/null || true

.PHONY: build
build:		cleanall all

.PHONY: dep
dep:		
		$(CXX) $(INCLUDES) $(CDEPFLAGS) $(SRC_FILES) | \
		sed 's/\(.*\)\.o:/$$(TMP_DIR)\/\1\.o:/' > $(DEP_FILE)

$(TMP_DIR)/%.o:	%.cpp	
		$(CXX) $(CXXFLAGS) $(DEFS) $(INCLUDES) -c $< -o $@

$(TMP_DIR)/%.o:	$(TMP_DIR)/%.cpp
		$(CXX) $(CXXFLAGS) $(DEFS) $(INCLUDES) -c $< -o $@

-include $(DEP_FILE)
		
$(OUTPUT):	$(OBJ_FILES) $(M4DLIBS)
		$(CXX) $(CXXFLAGS) $(OBJ_FILES) $(LIBDIRS) $(LIBS) -o $(OUTPUT) 

.PHONY: clean
clean:
		$(RM) $(OBJ_FILES) $(DEP_FILE)
		$(RM) $(OUTPUT)
