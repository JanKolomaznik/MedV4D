
MAKE=make


.PHONY:all
all:	cellCommon cellClient cellServer

.PHONY:clean-all
clean-all:	clean-cellCommon clean-cellClient clean-cellServer
#------------------------------------------------------------------------------
.PHONY:cellCommon
cellCommon:
	 $(MAKE) --directory=$@

.PHONY:cellClient
cellClient:
	 $(MAKE) --directory=$@

.PHONY:cellServer
cellServer:
	$(MAKE) --directory=$@
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
.PHONY:clean-cellCommon
clean-cellCommon:
	 $(MAKE) --directory=cellCommon clean

.PHONY:clean-cellClient
clean-cellClient:
	 $(MAKE) --directory=cellClient clean

.PHONY:clean-cellServer
clean-cellServer:
	 $(MAKE) --directory=cellServer clean
#------------------------------------------------------------------------------
