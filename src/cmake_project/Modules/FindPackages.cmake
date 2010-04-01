
set(Boost_USE_STATIC_LIBS   ON)
set(Boost_USE_MULTITHREADED ON)
FIND_PACKAGE(Boost REQUIRED COMPONENTS filesystem system thread)
FIND_PACKAGE(OpenGL REQUIRED)

INCLUDE( "${MEDV4D_CMAKE_MODULES_DIR}/Functions.cmake" )
INCLUDE( "${MEDV4D_CMAKE_MODULES_DIR}/FindQt4ForM4D.cmake" )
INCLUDE( "${MEDV4D_CMAKE_MODULES_DIR}/FindVTKForM4D.cmake" )
INCLUDE( "${MEDV4D_CMAKE_MODULES_DIR}/FindDCMTKForM4D.cmake" )

#Do some checking
SET( CG_SHADERS Cg CgGL )

IF(UNIX)
	SET(ADDITIONAL_LIBRARIES ${ADDITIONAL_LIBRARIES} wrap)
ENDIF(UNIX)

IF(WIN32)
	#Visual studio is able to link those automatically
	SET(Boost_LIBRARIES "")
ENDIF(WIN32)

SET( LIB_DIRS_3RD_PARTY ${VTK_LIBRARY_DIRS} ${DCMTK_LIBRARY_DIRS} ${Boost_LIBRARY_DIRS} ${QT_LIBRARY_DIR} )
SET( LIBRARIES_3RD_PARTY ${VTK_LIBRARIES} ${DCMTK_LIBRARIES} ${QT_LIBRARIES} ${ADDITIONAL_LIBRARIES} ${OPENGL_LIBRARY} ${Boost_LIBRARIES} ${CG_SHADERS} )
SET( INCLUDE_DIRS_3RD_PARTY ${VTK_INCLUDE_DIRS} ${DCMTK_INCLUDE_DIR} ${Boost_INCLUDE_DIRS} ${QT_INCLUDES} )

SET( MEDV4D_LIB_DIRS ${LIBRARY_OUTPUT_PATH} )
SET( MEDV4D_LIBRARIES m4dWidgets vtkIntegration backendForDICOM Imaging Common)
SET( MEDV4D_INCLUDE_DIRS ${MEDV4D_CMAKE_SOURCE_DIR}/.. ${MEDV4D_CMAKE_SOURCE_DIR}/../include )

SET( MEDV4D_ALL_LIB_DIRS  ${LIB_DIRS_3RD_PARTY} ${MEDV4D_LIB_DIRS} )
SET( MEDV4D_ALL_LIBRARIES  ${MEDV4D_LIBRARIES} ${LIBRARIES_3RD_PARTY} )
SET( MEDV4D_ALL_INCLUDE_DIRS ${INCLUDE_DIRS_3RD_PARTY} ${MEDV4D_INCLUDE_DIRS} )

INCLUDE_DIRECTORIES( ${MEDV4D_ALL_INCLUDE_DIRS} )
LINK_DIRECTORIES( ${MEDV4D_ALL_LIB_DIRS} )

