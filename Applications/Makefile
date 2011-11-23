
MAKE=make


.PHONY:all
all:	m4dPilot BoneSegmentation SimpleMIP tools

.PHONY:clean-all
clean-all:	clean-m4dPilot clean-BoneSegmentation clean-SimpleMIP clean-tools
#------------------------------------------------------------------------------
.PHONY:m4dPilot
m4dPilot:
	 $(MAKE) --directory=$@

.PHONY:BoneSegmentation
BoneSegmentation:
	 $(MAKE) --directory=$@

.PHONY:SimpleMIP
SimpleMIP:
	$(MAKE) --directory=$@

.PHONY:tools
tools:
	$(MAKE) --directory=$@
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
.PHONY:clean-m4dPilot
clean-m4dPilot:
	 $(MAKE) --directory=m4dPilot clean

.PHONY:clean-FilteringTest
clean-BoneSegmentation:
	 $(MAKE) --directory=BoneSegmentation clean

.PHONY:clean-SimpleMIP
clean-SimpleMIP:
	 $(MAKE) --directory=SimpleMIP clean

.PHONY:clean-tools
clean-tools:
	 $(MAKE) --directory=tools clean
#------------------------------------------------------------------------------
