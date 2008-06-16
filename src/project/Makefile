

MAKE=make


.PHONY:all
all:	libs Applications
	#no command

.PHONY:libs
libs:	common backendForDICOM Imaging GUI vtkIntegration
	#no command

.PHONY:clean-all
clean-all:	clean-libs clean-Applications
	#no command

.PHONY:clean-libs
clean-libs:	clean-common clean-backendForDICOM clean-Imaging clean-GUI clean-vtkIntegration
	#no command
#------------------------------------------------------------------------------
.PHONY:common
common:
	 $(MAKE) --directory=../$@

.PHONY:Imaging
Imaging:
	 $(MAKE) --directory=../$@

.PHONY:vtkIntegration
vtkIntegration:
	 $(MAKE) --directory=../$@

.PHONY:backendForDICOM
backendForDICOM:
	 $(MAKE) --directory=../$@

.PHONY:GUI
GUI:
	 $(MAKE) --directory=../$@

.PHONY:Applications
Applications:
	 $(MAKE) --directory=../$@

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
.PHONY:clean-common
clean-common:
	 $(MAKE) --directory=../common clean

.PHONY:clean-Imaging
clean-Imaging:
	 $(MAKE) --directory=../Imaging clean

.PHONY:clean-vtkIntegration
clean-vtkIntegration:
	 $(MAKE) --directory=../vtkIntegration clean

.PHONY:clean-backendForDICOM
clean-backendForDICOM:
	 $(MAKE) --directory=../backendForDICOM clean

.PHONY:clean-GUI
clean-GUI:
	 $(MAKE) --directory=../GUI clean-all

.PHONY:clean-Applications
clean-Applications:
	 $(MAKE) --directory=../Applications clean-all

#------------------------------------------------------------------------------
