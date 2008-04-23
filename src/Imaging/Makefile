
srcdir = .

CC=g++
AR=ar

NAME=		Imaging
OUTPUTNAME=	lib$(NAME).a
OUTPUTDIR=	../lib
OUTPUT=		$(OUTPUTDIR)/$(OUTPUTNAME)
TMP_DIR=	../tmp/$(NAME)


CCFLAGS=	-Wall
DEP_FILE=	$(TMP_DIR)/depend
CDEPFLAGS=	-MM -MF$(DEP_FILE)
LDFLAGS=
ARFLAGS=	-r
INCLUDES=	-I$(srcdir)\
		-I../include


DEFS=

DEBUG_DEFS=	-DDEBUG_LEVEL=10\
		-DDEBUG_ADITIONAL_INFO


TARGETS=	AbstractImage \
		ImageFactory

SRC_FILES=	$(addsuffix .cpp, $(TARGETS))
OBJ_FILES=	$(addprefix $(TMP_DIR)/, $(addsuffix .o, $(TARGETS)) )

.PHONY: all

all:		dep $(OUTPUT)

.PHONY: dep
dep:
		$(CC) $(INCLUDES) $(CDEPFLAGS) $(SRC_FILES)

-include $(DEP_FILE)

$(OBJ_FILES):	
		$(CC) $(CCFLAGS) $(INCLUDES) -c $(patsubst $(TMP_DIR)/%.o,%.cpp, $@ ) -o $@
		
$(OUTPUT):	$(OBJ_FILES)
		$(AR) $(ARFLAGS) $(OUTPUT) $(OBJ_FILES)

.PHONY: clean
clean:
		rm -f $(TMP_DIR)/*.o $(DEP_FILE) $(OUTPUT)
