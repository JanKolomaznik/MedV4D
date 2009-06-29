srcTop=..

.PHONY:all
all:	libs Applications

.PHONY:libs
libs:	common backendForDICOM Imaging GUI vtkIntegration

.PHONY:clean-all
clean-all:	clean-libs clean-Applications
	#no command

.PHONY:clean-libs
clean-libs:	clean-common clean-backendForDICOM clean-Imaging clean-GUI clean-vtkIntegration
	#no command
#------------------------------------------------------------------------------
.PHONY:common
common:
	 $(MAKE) --directory=$(srcTop)/$@ all

.PHONY:Imaging
Imaging:
	 $(MAKE) --directory=$(srcTop)/$@ all

.PHONY:vtkIntegration
vtkIntegration:
	 $(MAKE) --directory=$(srcTop)/$@ all

.PHONY:backendForDICOM
backendForDICOM:
	 $(MAKE) --directory=$(srcTop)/$@ all

.PHONY:GUI
GUI:
	 $(MAKE) --directory=$(srcTop)/$@ all

.PHONY:commonRCPart
commonRCPart:
	$(MAKE) --directory=$(srcTop)/remoteComp/commonPart all

.PHONY:clientRCPart
clientRCPart:
	$(MAKE) --directory=$(srcTop)/remoteComp/clientPart all

.PHONY:serverRCPart
serverRCPart:
	$(MAKE) --directory=$(srcTop)/remoteComp/serverPart all

.PHONY:Applications
Applications:
	 $(MAKE) --directory=$(srcTop)/$@ all

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
.PHONY:clean-common
clean-common:
	 $(MAKE) --directory=$(srcTop)/common clean

.PHONY:clean-Imaging
clean-Imaging:
	 $(MAKE) --directory=$(srcTop)/Imaging clean

.PHONY:clean-vtkIntegration
clean-vtkIntegration:
	 $(MAKE) --directory=$(srcTop)/vtkIntegration clean

.PHONY:clean-backendForDICOM
clean-backendForDICOM:
	 $(MAKE) --directory=$(srcTop)/backendForDICOM clean

.PHONY:clean-GUI
clean-GUI:
	 $(MAKE) --directory=$(srcTop)/GUI clean-all

.PHONY:clean-commonRCPart
clean-commonRCPart:
	$(MAKE) --directory=$(srcTop)/remoteComp/commonPart clean

.PHONY:clean-clientRCPart
clean-clientRCPart:
	$(MAKE) --directory=$(srcTop)/remoteComp/clientPart clean

.PHONY:clean-serverRCPart
clean-serverRCPart:
	$(MAKE) --directory=$(srcTop)/remoteComp/serverPart clean

.PHONY:clean-Applications
clean-Applications:
	 $(MAKE) --directory=$(srcTop)/Applications clean-all

#------------------------------------------------------------------------------
