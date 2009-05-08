# --------------------------------------------------------------- 
# Licensed Materials - Property of IBM                            
# 5724-S84                                                        
# (C) Copyright IBM Corp. 2008       All Rights Reserved          
# US Government Users Restricted Rights - Use, duplication or     
# disclosure restricted by GSA ADP Schedule Contract with         
# IBM Corp.                                                       
# --------------------------------------------------------------- 
# PROLOG END TAG zYx

CELL_TOP=/opt/cell/sdk
srcTop=../../../../..

include $(srcTop)/project/ToolchainSelect.mk
include $(srcTop)/project/pathsDefinitions.mk
include $(srcTop)/project/configurationSelector.mk

TARGET_PROCESSOR := ppu64

########################################################################
#			Target
########################################################################

LIBRARY	:= libPPEStuff$(archPostfix).a

# path to source files
VPATH := src

########################################################################
#			Local Defines
########################################################################
#--param max-unroll-times=1 # needed to keep size of program down
#MY_SPECIAL_FLAGS = -ffunction-sections -fdata-sections -g
#CFLAGS_gcc:= CXXFLAGS
#LDFLAGS_gcc= --gc-sections --print-gc-sections --stack-analysis --local-store=0:256	#-Wl,-q -g
				
INCLUDE         := -I$(srcTop)\
					$(ITKIncludes)

INSTALL_DIR		= $(srcTop)/lib
INSTALL_FILES		= $(LIBRARY)

########################################################################
#			buildutils/make.footer
########################################################################

ifdef CELL_TOP
	include $(CELL_TOP)/buildutils/make.footer
else
	include ../../../../../buildutils/make.footer
endif