

MAKE=make

MAKE_ATR=

.PHONY:all
all:	common

.PHONY:all-clean
all-clean: clean_set all

.PHONY:common
common:
	 cd ../common && $(MAKE) $(MAKE_ATR) 

.PHONY: clean_set
clean_set: 
	MAKE_ATR:=-cleanall

