
MAKE=make


.PHONY:all
all:	m4dPilot BoneSegmentation

.PHONY:clean-all
clean-all:	clean-m4dPilot clean-BoneSegmentation
#------------------------------------------------------------------------------
.PHONY:m4dPilot
m4dPilot:
	 $(MAKE) --directory=$@

.PHONY:BoneSegmentation
BoneSegmentation:
	 $(MAKE) --directory=$@


#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
.PHONY:clean-m4dPilot
clean-m4dPilot:
	 $(MAKE) --directory=m4dPilot clean

.PHONY:clean-FilteringTest
clean-BoneSegmentation:
	 $(MAKE) --directory=BoneSegmentation clean

#------------------------------------------------------------------------------
