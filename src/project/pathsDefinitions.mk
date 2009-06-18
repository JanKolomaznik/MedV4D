
##########################################
ifeq "$(ARCH)" "CellCross"

# path to ITK libraries and includes
ITKIncludeDir=/usr/local/include/InsightToolkit
ifeq "$(CONF)" "Release"
ITKLibsDir=/data/cell/ITK/CellRelease
else
ITKLibsDir=/data/cell/ITK/LIBCell/bin
endif

# path to VTK libraries & includes
VTKLibsDir=/usr/local/lib/vtk-5.0
VTKIncludeDir=/usr/local/include/vtk-5.0
endif
##########################################
ifeq "$(ARCH)" "CellPCTest"
# path to ITK libraries and includes
ITKIncludeDir=/usr/local/include/InsightToolkit
ITKLibsDir=/data/cell/ITK/LIBCell/bin

# path to VTK libraries & includes
VTKLibsDir=/usr/local/lib/vtk-5.0
VTKIncludeDir=/usr/local/include/vtk-5.0
endif
##########################################
ifeq "$(ARCH)" "CellNative"

# path to ITK libraries and includes
ITKIncludeDir=/usr/local/include/InsightToolkit
ITKLibsDir=/data/cell/ITK/LIBCell/bin

# path to VTK libraries & includes
VTKLibsDir=/usr/local/lib/vtk-5.0
VTKIncludeDir=/usr/local/include/vtk-5.0
endif
##########################################
ifeq "$(ARCH)" "PC"
# path to ITK libraries and includes
ITKIncludeDir=/usr/local/include/InsightToolkit

ifeq "$(CONF)" "Release"
#TODO
ITKLibsDir=/data/cell/ITK/LIBCell/bin
endif
ITKLibsDir=/usr/local/lib/InsightToolkit

# path to VTK libraries & includes
VTKLibsDir=/usr/local/lib/vtk-5.0
VTKIncludeDir=/usr/local/include/vtk-5.0
endif
##########################################

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

DCMTK_INCLUDE_PATH := /usr/local/dicom/include
DCMTK_LIB_PATH := /usr/local/dicom/lib
DCMTKLIBS=	-ldcmnet -ldcmdata -lofstd