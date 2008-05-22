

MAKE=make


.PHONY:all
all:	libs Applications
	#no command

.PHONY:libs
libs:	common backendForDicom Imaging GUI
	#no command

.PHONY:clean-all
clean-all:	clean-libs clean-Applications
	#no command

.PHONY:clean-libs
clean-libs:	clean-common clean-backendForDicom clean-Imaging clean-GUI
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

.PHONY:backendForDicom
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

.PHONY:clean-backendForDicom
clean-backendForDICOM:
	 $(MAKE) --directory=../backendForDicom clean

.PHONY:clean-GUI
clean-GUI:
	 $(MAKE) --directory=../GUI clean-all

.PHONY:clean-Applications
clean-Applications:
	 $(MAKE) --directory=../Applications clean-all

#------------------------------------------------------------------------------
