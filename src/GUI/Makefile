
MAKE=make


.PHONY:all
all:	widgets

.PHONY:clean-all
clean-all:	clean-widgets
#------------------------------------------------------------------------------
.PHONY:widgets
widgets:
	 $(MAKE) --directory=$@

#.PHONY:selectors
#selectors:
#	 $(MAKE) --directory=$@
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
.PHONY:clean-widgets
clean-widgets:
	 $(MAKE) --directory=widgets clean

#.PHONY:clean-selectors
#clean-selectors:
#	 $(MAKE) --directory=selectors clean
#------------------------------------------------------------------------------
