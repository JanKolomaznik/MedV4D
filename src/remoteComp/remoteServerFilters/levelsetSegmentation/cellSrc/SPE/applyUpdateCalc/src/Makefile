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
srcTop=../../../../../../..

########################################################################
#			Target
########################################################################

LIBRARY_spu := applyUpdateCalc.a

########################################################################
#			Local Defines
########################################################################
#-ffunction-sections -fdata-sections #-g #--param max-unroll-times=1 # needed to keep size of program down
CFLAGS_gcc:= -g -DFOR_CELL
LDFLAGS_gcc= --gc-sections --print-gc-sections --stack-analysis --local-store=0:256	#-Wl,-q -g
				
INCLUDE         := -I$(srcTop)

########################################################################
#			buildutils/make.footer
########################################################################

include $(CELL_TOP)/buildutils/make.footer
