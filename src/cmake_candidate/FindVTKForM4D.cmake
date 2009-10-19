IF( ${WIN32} )
	SET( WIN32_USE_PREPARED_PACKAGES_VTK TRUE )
ENDIF( ${WIN32} )



IF( WIN32_USE_PREPARED_PACKAGES_VTK )
	
	SET(VTK_Components vtkCommon vtkDICOMParser vtkexoIIc vtkexpat 
	vtkFiltering vtkfreetype vtkftgl vtkGenericFiltering vtkGraphics 
	vtkHybrid vtkImaging vtkIO vtkjpeg vtkNetCDF vtkpng vtkRendering 
	vtksys vtktiff vtkVolumeRendering vtkWidgets vtkzlib QVTK ) 

	# The VTK include file directories.
	SET(VTK_INCLUDE_DIRS "${CMAKE_SOURCE_DIR}/../include/vtk")

	# The VTK library directories.
	SET(VTK_LIBRARY_DIRS "${CMAKE_SOURCE_DIR}/../lib/vtk")
	SET(VTK_LIBRARY_DIR "${CMAKE_SOURCE_DIR}/../lib/vtk")

	# The VTK version number
	SET(VTK_MAJOR_VERSION "5")
	SET(VTK_MINOR_VERSION "0")
	SET(VTK_BUILD_VERSION "4")

	PREPEND_STRING_TO_LIST_MEMBERS( "${VTK_LIBRARY_DIR}/"  VTK_Components VTK_LIBS_OPTIMIZED )
	APPEND_STRING_TO_LIST_MEMBERS( ".lib"  VTK_LIBS_OPTIMIZED VTK_LIBS_OPTIMIZED )
	INSERT_KEYWORD_BEFORE_EACH_MEMBER( "optimized" VTK_LIBS_OPTIMIZED VTK_LIBS_OPTIMIZED )
	
	PREPEND_STRING_TO_LIST_MEMBERS( "${VTK_LIBRARY_DIR}/"  VTK_Components VTK_LIBS_DEBUG )
	APPEND_STRING_TO_LIST_MEMBERS( "d.lib"  VTK_LIBS_DEBUG VTK_LIBS_DEBUG )
	INSERT_KEYWORD_BEFORE_EACH_MEMBER( "debug" VTK_LIBS_DEBUG VTK_LIBS_DEBUG )
	
	SET( VTK_LIBRARIES ${VTK_LIBS_DEBUG} ${VTK_LIBS_OPTIMIZED} )
	
ELSE( WIN32_USE_PREPARED_PACKAGES_VTK )
	SET(VTK_Components vtkCommon vtkDICOMParser vtkexoIIc 
	vtkFiltering vtkftgl vtkGenericFiltering vtkGraphics 
	vtkHybrid vtkImaging vtkIO vtkNetCDF vtkRendering 
	vtksys vtkVolumeRendering vtkWidgets QVTK ) 

	FIND_PACKAGE(VTK REQUIRED)
	INCLUDE( ${VTK_USE_FILE} )
	SET( VTK_LIBRARIES ${VTK_Components} )
ENDIF( WIN32_USE_PREPARED_PACKAGES_VTK )

#message( "${VTK_LIBRARIES}" )
#message( "${VTK_USE_FILE}" )
