
IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Debug CACHE STRING
      "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
      FORCE)
ENDIF(NOT CMAKE_BUILD_TYPE)

IF(NOT MEDV4D_CMAKE_SOURCE_DIR)
	MESSAGE( FATAL_ERROR "MEDV4D_CMAKE_SOURCE_DIR has to be set!!!" )
ENDIF(NOT MEDV4D_CMAKE_SOURCE_DIR)

SET(MEDV4D_CMAKE_MODULES_DIR ${MEDV4D_CMAKE_SOURCE_DIR}/Modules)
SET(CMAKE_MODULE_PATH "${MEDV4D_CMAKE_MODULES_DIR}" "${CMAKE_ROOT}/Modules" )

INCLUDE( "${MEDV4D_CMAKE_MODULES_DIR}/Functions.cmake" )
INCLUDE( "${MEDV4D_CMAKE_MODULES_DIR}/FindPackages.cmake" )
INCLUDE( "${MEDV4D_CMAKE_MODULES_DIR}/StandardCompileOptions.cmake" )


#Mandatory -- prevents adding targets for Medv4DLibs more than once 
IF(NOT MEDV4D_LIBS_CMAKE_PROCESSED)
	SET( MEDV4D_TMP_DIR "${MEDV4D_CMAKE_SOURCE_DIR}/../tmp/${CMAKE_BUILD_TYPE}" )
	ADD_SUBDIRECTORY( "${MEDV4D_CMAKE_SOURCE_DIR}/Medv4DLibs" ${MEDV4D_TMP_DIR} )
	#Must propagate to parent scope
	SET( MEDV4D_LIBS_CMAKE_PROCESSED TRUE PARENT_SCOPE)
ENDIF(NOT MEDV4D_LIBS_CMAKE_PROCESSED)

ADD_DEFINITIONS( ${MEDV4D_COMPILE_DEFINITIONS} )
