
##########################################
ifeq "$(ARCH)" "CellCross"

# path to ITK libraries
ifeq "$(CONF)" "Release"
ITKLibsDir=$(srcTop)/lib/3rdParty/Release/ITK.ppc64
else
ITKLibsDir=$(srcTop)/lib/3rdParty/Debug/ITK.ppc64
endif

INCLUDES += -I$(srcTop)/include/3rdParty/asio-devel.ppc64

endif
##########################################
ifeq "$(ARCH)" "Cell_PCSimulation"
# path to ITK libraries
ITKLibsDir=$(srcTop)/lib/3rdParty/Debug/ITK.i686

endif
##########################################
ifeq "$(ARCH)" "CellPCTest"
# path to ITK libraries
ITKLibsDir=$(srcTop)/lib/3rdParty/Debug/ITK.i686

endif
##########################################
ifeq "$(ARCH)" "CellNative"

# path to ITK libraries
ifeq "$(CONF)" "Release"
ITKLibsDir=$(srcTop)/lib/3rdParty/Release/ITK.ppc64
else
ITKLibsDir=$(srcTop)/lib/3rdParty/Debug/ITK.ppc64
endif

endif
##########################################
ifeq "$(ARCH)" "PC"
# path to ITK libraries

ifeq "$(CONF)" "Release"
ITKLibsDir=$(srcTop)/lib/3rdParty/Release/ITK.i686
else
ITKLibsDir=$(srcTop)/lib/3rdParty/Debug/ITK.i686
#ITKLibsDir=/usr/local/lib/InsightToolkit
endif

endif
##########################################

# path to VTK libraries & includes
VTKLibsDir=	-L/usr/local/lib/vtk-5.0
VTKIncludeDir=	-I/usr/include/vtk\
		-I/usr/local/include/vtk-5.0

ITKIncludeDir=$(srcTop)/include/3rdParty/ITK.noarch
ITKIncludes=	-I$(ITKIncludeDir)\
		-I$(ITKIncludeDir)/Common\
		-I$(ITKIncludeDir)/Algorithms\
		-I$(ITKIncludeDir)/BasicFilters\
		-I$(ITKIncludeDir)/IO\
		-I$(ITKIncludeDir)/Utilities\
		-I$(ITKIncludeDir)/Utilities/vxl/vcl\
		-I$(ITKIncludeDir)/Utilities/vxl/core\
		-I$(ITKIncludeDir)/gdcm/src

ITKLIBS= 	$(ITKLibsDir)/libITKNumerics.a\
		$(ITKLibsDir)/libITKSpatialObject.a\
		$(ITKLibsDir)/libITKCommon.a\
		$(ITKLibsDir)/libitkvnl_inst.a\
		$(ITKLibsDir)/libitkvnl_algo.a\
		$(ITKLibsDir)/libitkvnl.a\
		$(ITKLibsDir)/libitkvcl.a\
		$(ITKLibsDir)/libitksys.a\
		$(ITKLibsDir)/libitkv3p_netlib.a
		#$(ITKLibsDir)/libITKNrrdIO.a\
		#$(ITKLibsDir)/libITKMetaIO.a\
		#$(ITKLibsDir)/libITKEXPAT.a\
		#$(ITKLibsDir)/libITKniftiio.a\
		#$(ITKLibsDir)/libITKznz.a\

VTKLIBS=	-lvtkCommon\
		-lvtkDICOMParser\
		-lvtkexoIIc\
		-lvtkFiltering\
		-lvtkftgl\
		-lvtkGenericFiltering\
		-lvtkGraphics\
		-lvtkHybrid\
		-lvtkImaging\
		-lvtkIO\
		-lvtkNetCDF\
		-lvtkRendering\
		-lvtksys\
		-lvtkVolumeRendering\
		-lvtkWidgets\
		-lQVTK\
		-lwrap

		#-lvtkexpat\
		#-lvtkfreetype\
		#-lvtkjpeg\
		#-lvtkpng\
		#-lvtktiff\
		#-lvtkzlib

QTIncludDirs := -I/usr/include/qt4\
		-I/usr/include/qt4/Qt\
		-I/usr/include/qt4/QtCore\
		-I/usr/include/qt4/QtGui\
		-I/usr/include/qt4/QtOpenGL\
		-I/usr/include/Qt\
		-I/usr/include/QtGui\
		-I/usr/include/QtCore

QTLibDirs := -L/usr/lib/qt4

QTLIBS=		-lQtCore\
		-lQtGui\
		-lQtOpenGL

DCMTK_INCLUDE_PATH := $(srcTop)/include/3rdParty/dcmtk.noarch
DCMTK_LIB_PATH := $(srcTop)/lib/3rdParty/Debug/dcmtk.i686
DCMTKLIBS=	-ldcmnet -ldcmdata -lofstd