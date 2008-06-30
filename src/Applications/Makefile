
MAKE=make


.PHONY:all
all:	m4dPilot FilteringTest ViewerTest

.PHONY:clean-all
clean-all:	clean-m4dPilot clean-FilteringTest clean-ViewerTest
#------------------------------------------------------------------------------
.PHONY:m4dPilot
m4dPilot:
	 $(MAKE) --directory=$@

.PHONY:FilteringTest
FilteringTest:
	 $(MAKE) --directory=$@

.PHONY:ViewerTest
ViewerTest:
	 $(MAKE) --directory=$@
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
.PHONY:clean-m4dPilot
clean-m4dPilot:
	 $(MAKE) --directory=m4dPilot clean

.PHONY:clean-FilteringTest
clean-FilteringTest:
	 $(MAKE) --directory=FilteringTest clean

.PHONY:clean-ViewerTest
clean-ViewerTest:
	 $(MAKE) --directory=ViewerTest clean
#------------------------------------------------------------------------------
