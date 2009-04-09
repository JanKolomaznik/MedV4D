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

########################################################################
#			Target
########################################################################

#PROGRAMS_spu 	:= updateCalculatorSPEMain

LIBRARY_spu := updateSolver.a

#LIBRARY_embed64	:= updateCalculatorSPEMain.a


########################################################################
#			Local Defines
########################################################################

CFLAGS_gcc:= -g $(PLATFORM_DEFS) -ffunction-sections -fdata-sections #-g #--param max-unroll-times=1 # needed to keep size of program down
LDFLAGS_gcc= --gc-sections --print-gc-sections --stack-analysis --local-store=0:256	#-Wl,-q -g

srcTop=../../../../../..
				
INCLUDE         := -I$(srcTop)

#INSTALL_DIR		= $(srcTop)/SPU
#INSTALL_FILES		= $(LIBRARY_embed64)

########################################################################
#			buildutils/make.footer
########################################################################

ifdef CELL_TOP
	include $(CELL_TOP)/buildutils/make.footer
else
	include ../../../../../buildutils/make.footer
endif