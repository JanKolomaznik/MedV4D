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
#tools/src 
DIRS	:= 	updateCalculation/src applyUpdateCalc/src

########################################################################
#			Target
########################################################################

LIBRARY_embed64	:= libSPEPart$(archPostfix).a
PROGRAMS_spu 	:= SPEMain

########################################################################
#			Local Defines
########################################################################

MY_SPECIAL_FLAGS = -ffunction-sections -fdata-sections #-g #--param max-unroll-times=1 # needed to keep size of program down
CFLAGS_gcc:= -g -DCOMPILE_FOR_CELL
#LDFLAGS_gcc= --gc-sections --print-gc-sections --stack-analysis --local-store=0:256	#-Wl,-q -g
				
INCLUDE         := -I$(srcTop)

IMPORTS := updateCalculation/src/updateSolver.a applyUpdateCalc/src/applyUpdateCalc.a

INSTALL_DIR		= $(srcTop)/lib/SPU
INSTALL_FILES		= $(LIBRARY_embed64)

########################################################################
#			buildutils/make.footer
########################################################################

include $(CELL_TOP)/buildutils/make.footer
	