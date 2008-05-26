
MAKE=make


.PHONY:all
all:	m4dPilot FilteringTest

.PHONY:clean-all
clean-all:	clean-m4dPilot clean-FilteringTest
#------------------------------------------------------------------------------
.PHONY:m4dPilot
m4dPilot:
	 $(MAKE) --directory=$@

.PHONY:FilteringTest
FilteringTest:
	 $(MAKE) --directory=$@
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
.PHONY:clean-m4dPilot
clean-m4dPilot:
	 $(MAKE) --directory=m4dPilot clean

.PHONY:clean-FilteringTest
clean-FilteringTest:
	 $(MAKE) --directory=FilteringTest clean
#------------------------------------------------------------------------------
