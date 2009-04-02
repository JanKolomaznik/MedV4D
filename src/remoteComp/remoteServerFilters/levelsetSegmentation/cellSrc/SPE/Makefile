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

PROGRAMS_spu 	:= updateCalculatorSPE

LIBRARY_embed	:= updateCalculatorSPE.a


########################################################################
#			Local Defines
########################################################################

srcTop=../../../../..

ITKIncludeRoot=/usr/local/include/InsightToolkit
ITKIncludes=	-I$(ITKIncludeRoot)\
				-I$(ITKIncludeRoot)/Common\
				-I$(ITKIncludeRoot)/Algorithms\
				-I$(ITKIncludeRoot)/BasicFilters\
				-I$(ITKIncludeRoot)/IO\
				-I$(ITKIncludeRoot)/Utilities\
				-I$(ITKIncludeRoot)/Utilities/vxl/vcl\
				-I$(ITKIncludeRoot)/Utilities/vxl/core\
				-I$(ITKIncludeRoot)/gdcm/src
				
INCLUDE         := -I$(srcTop) $(ITKIncludes)

########################################################################
#			buildutils/make.footer
########################################################################

ifdef CELL_TOP
	include $(CELL_TOP)/buildutils/make.footer
else
	include ../../../../../buildutils/make.footer
endif