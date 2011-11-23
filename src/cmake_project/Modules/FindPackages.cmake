
if(POLICY CMP0012)
	cmake_policy(SET CMP0012 NEW)
endif(POLICY CMP0012)
if(POLICY CMP0011)
	cmake_policy(SET CMP0011 NEW)
endif(POLICY CMP0011)


INCLUDE( "${MEDV4D_CMAKE_MODULES_DIR}/Functions.cmake" )
INCLUDE( "${MEDV4D_CMAKE_MODULES_DIR}/FindQt4ForM4D.cmake" )
INCLUDE( "${MEDV4D_CMAKE_MODULES_DIR}/StandardCompileOptions.cmake" )

IF( ${PYTHON_ENABLED} MATCHES "ON" )
	#Do some checking
	SET(BOOST_PYTHON python)

	FIND_PACKAGE( PythonLibs REQUIRED ) #TODO  handle release/debug
ENDIF( ${PYTHON_ENABLED} MATCHES "ON" )

SET(Boost_USE_MULTITHREADED ON)
FIND_PACKAGE(Boost REQUIRED COMPONENTS filesystem system thread ${BOOST_PYTHON})

FIND_PACKAGE(OpenGL REQUIRED)

IF( ${VTK_ENABLED} MATCHES "ON" )
	INCLUDE( "${MEDV4D_CMAKE_MODULES_DIR}/FindVTKForM4D.cmake" )
ENDIF( ${VTK_ENABLED} MATCHES "ON" )
INCLUDE( "${MEDV4D_CMAKE_MODULES_DIR}/FindDCMTKForM4D.cmake" )


SET(LIBRARY_OUTPUT_PATH ${MEDV4D_CMAKE_SOURCE_DIR}/../lib)
SET(EXECUTABLE_OUTPUT_PATH ${MEDV4D_CMAKE_SOURCE_DIR}/../executables)

IF( ${CG_ENABLED} MATCHES "ON" )
	#Do some checking
	FIND_PACKAGE( Cg REQUIRED )
	SET( CG_SHADER_LIBRARIES ${CG_LIBRARY} ${CG_GL_LIBRARY} )
	SET( CG_SHADER_LIBRARY_DIRS "" )
	SET( CG_SHADER_INCLUDE_DIRS ${CG_INCLUDE_PATH} )
	#MESSAGE( STATUS "CG libs: ${CG_SHADER_LIBRARIES}"
ENDIF( ${CG_ENABLED} MATCHES "ON" )

IF( ${DEVIL_ENABLED} MATCHES "ON" )
	#Do some checking
	FIND_PACKAGE( DevIL REQUIRED )
	SET( DEVIL_LIBRARIES ${DEVIL_LIB_IL} ${DEVIL_LIB_ILU} )
	SET( DEVIL_LIBRARY_DIRS "" )
	SET( DEVIL_INCLUDE_DIRS ${IL_INCLUDE_DIR} )
	#MESSAGE( STATUS "DevIL libs: ${DEVIL_LIBRARIES}"
ENDIF( ${DEVIL_ENABLED} MATCHES "ON" )

IF(UNIX)
	SET(ADDITIONAL_LIBRARIES ${ADDITIONAL_LIBRARIES} wrap )
ENDIF(UNIX)

IF(WIN32)
	#Visual studio is able to link those automatically
	#TODO - add test for win compiler
	SET(Boost_LIBRARIES "")
	
	SET(ADDITIONAL_LIBRARIES ${ADDITIONAL_LIBRARIES} glew32s Imm32 Winmm ws2_32)
	SET(ADDITIONAL_LIBRARY_DIR ${ADDITIONAL_LIBRARY_DIR} ${MEDV4D_CMAKE_SOURCE_DIR}/../lib/other)
	ADD_DEFINITIONS( -DGLEW_STATIC )
ELSE(WIN32)
	SET(ADDITIONAL_LIBRARIES ${ADDITIONAL_LIBRARIES} GLEW)
	#SET(ADDITIONAL_LIBRARY_DIR ${ADDITIONAL_LIBRARY_DIR} "/usr/lib/x86_64-linux-gnu/")
ENDIF(WIN32)

IF(USE_TBB)
	SET(ADDITIONAL_LIBRARIES ${ADDITIONAL_LIBRARIES} tbb tbbmalloc)
ENDIF(USE_TBB)


SET( LIB_DIRS_3RD_PARTY ${VTK_LIBRARY_DIRS} ${DCMTK_LIBRARY_DIRS} ${Boost_LIBRARY_DIRS} ${QT_LIBRARY_DIR} ${CG_SHADER_LIBRARY_DIRS} 
			${DEVIL_LIBRARY_DIRS} ${ADDITIONAL_LIBRARY_DIR}  )
SET( LIBRARIES_3RD_PARTY ${VTK_LIBRARIES} ${DCMTK_LIBRARIES} ${QT_LIBRARIES} ${ADDITIONAL_LIBRARIES} ${OPENGL_LIBRARY} ${Boost_LIBRARIES} ${CG_SHADER_LIBRARIES} 
			${DEVIL_LIBRARIES} ${PYTHON_LIBRARIES} ${PYTHON_DEBUG_LIBRARIES} )
SET( INCLUDE_DIRS_3RD_PARTY ${VTK_INCLUDE_DIRS} ${DCMTK_INCLUDE_DIR} ${Boost_INCLUDE_DIRS} ${QT_INCLUDES} ${CG_SHADER_INCLUDE_DIRS} 
			${OPENGL_INCLUDE_DIR} ${DEVIL_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS} )
MESSAGE( STATUS "VTK libs: ${VTK_LIBRARIES} ")
MESSAGE( STATUS "DCMTK libs: ${DCMTK_LIBRARIES} ")
MESSAGE( STATUS "QT libs: ${QT_LIBRARIES} ")
MESSAGE( STATUS "OpenGL libs: ${OPENGL_LIBRARY} ")
MESSAGE( STATUS "Boost libs: ${Boost_LIBRARIES} ")
MESSAGE( STATUS "CG libs: ${CG_SHADER_LIBRARIES} ")
MESSAGE( STATUS "DevIL libs: ${DEVIL_LIBRARIES}")
MESSAGE( STATUS "Python libs: ${PYTHON_LIBRARIES} ${PYTHON_DEBUG_LIBRARIES}" )
MESSAGE( STATUS "Additional libs: ${ADDITIONAL_LIBRARIES} ")

SET( MEDV4D_LIB_DIRS ${LIBRARY_OUTPUT_PATH} )
SET( MEDV4D_LIB_TARGETS m4dWidgets backendForDICOM Imaging Common)

IF( ${VTK_ENABLED} MATCHES "ON" )
	SET( MEDV4D_LIB_TARGETS vtkIntegration ${MEDV4D_LIB_TARGETS})
ENDIF( ${VTK_ENABLED} MATCHES "ON" )
SET( MEDV4D_INCLUDE_DIRS ${MEDV4D_CMAKE_SOURCE_DIR}/.. ${MEDV4D_CMAKE_SOURCE_DIR}/../include )

CREATE_LIB_NAMES_FROM_TARGET_NAMES( MEDV4D_LIB_TARGETS MEDV4D_LIBRARIES )
SET( MEDV4D_ALL_LIB_DIRS  ${LIB_DIRS_3RD_PARTY} ${MEDV4D_LIB_DIRS} )
SET( MEDV4D_ALL_LIBRARIES  ${MEDV4D_LIBRARIES} ${LIBRARIES_3RD_PARTY} )
SET( MEDV4D_ALL_INCLUDE_DIRS ${INCLUDE_DIRS_3RD_PARTY} ${MEDV4D_INCLUDE_DIRS} )

#MESSAGE(STATUS "Link libraries: ${MEDV4D_ALL_LIBRARIES}" )
INCLUDE_DIRECTORIES( ${MEDV4D_ALL_INCLUDE_DIRS} )
LINK_DIRECTORIES( ${MEDV4D_ALL_LIB_DIRS} )

