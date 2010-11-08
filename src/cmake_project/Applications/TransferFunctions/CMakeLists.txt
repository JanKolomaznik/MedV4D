CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
PROJECT( TransferFunctions )

IF(NOT MEDV4D_CMAKE_SOURCE_DIR)
	SET(MEDV4D_CMAKE_SOURCE_DIR "${CMAKE_SOURCE_DIR}/../.." )
	SET(MEDV4D_CMAKE_MODULES_DIR ${MEDV4D_CMAKE_SOURCE_DIR}/Modules)
	SET(CMAKE_MODULE_PATH "${MEDV4D_CMAKE_MODULES_DIR}" "${CMAKE_ROOT}/Modules" )
ENDIF(NOT MEDV4D_CMAKE_SOURCE_DIR)

INCLUDE( "${MEDV4D_CMAKE_MODULES_DIR}/Functions.cmake" )
INCLUDE( "${MEDV4D_CMAKE_MODULES_DIR}/FindPackages.cmake" )
INCLUDE( "${MEDV4D_CMAKE_MODULES_DIR}/StandardCompileOptions.cmake" )

#Mandatory -- prevents adding targets for Medv4DLibs more than once 
IF(NOT MEDV4D_LIBS_CMAKE_PROCESSED)
	ADD_SUBDIRECTORY( "${MEDV4D_CMAKE_SOURCE_DIR}/Medv4DLibs" "${MEDV4D_CMAKE_SOURCE_DIR}/../tmp" )
	#Must propagate to parent scope
	SET( MEDV4D_LIBS_CMAKE_PROCESSED TRUE PARENT_SCOPE)
ENDIF(NOT MEDV4D_LIBS_CMAKE_PROCESSED)

SET(EXECUTABLE_OUTPUT_PATH ${MEDV4D_CMAKE_SOURCE_DIR}/../executables)
TARGET_MEDV4D_PROGRAM( TransferFunctions ${MEDV4D_CMAKE_SOURCE_DIR}/../Applications/TransferFunctions )

#copy shader to project dir
configure_file(${MEDV4D_CMAKE_SOURCE_DIR}/../GUI/shaders/ImageRender.cgfx
    ${CMAKE_CURRENT_BINARY_DIR}/ImageRender.cgfx COPYONLY)

#copy libs to project debug dir
configure_file(${MEDV4D_CMAKE_SOURCE_DIR}/../bin/qt/QtCore4.dll
    ${MEDV4D_CMAKE_SOURCE_DIR}/../executables/Debug/QtCore4.dll COPYONLY)
configure_file(${MEDV4D_CMAKE_SOURCE_DIR}/../bin/qt/QtCored4.dll
    ${MEDV4D_CMAKE_SOURCE_DIR}/../executables/Debug/QtCored4.dll COPYONLY)
configure_file(${MEDV4D_CMAKE_SOURCE_DIR}/../bin/qt/QtGui4.dll
    ${MEDV4D_CMAKE_SOURCE_DIR}/../executables/Debug/QtGui4.dll COPYONLY)
configure_file(${MEDV4D_CMAKE_SOURCE_DIR}/../bin/qt/QtGuid4.dll
    ${MEDV4D_CMAKE_SOURCE_DIR}/../executables/Debug/QtGuid4.dll COPYONLY)
configure_file(${MEDV4D_CMAKE_SOURCE_DIR}/../bin/qt/QtOpenGL4.dll
    ${MEDV4D_CMAKE_SOURCE_DIR}/../executables/Debug/QtOpenGL4.dll COPYONLY)
configure_file(${MEDV4D_CMAKE_SOURCE_DIR}/../bin/qt/QtOpenGLd4.dll
    ${MEDV4D_CMAKE_SOURCE_DIR}/../executables/Debug/QtOpenGLd4.dll COPYONLY)

#copy libs to project debug dir
configure_file(${MEDV4D_CMAKE_SOURCE_DIR}/../bin/qt/QtCore4.dll
    ${MEDV4D_CMAKE_SOURCE_DIR}/../executables/Release/QtCore4.dll COPYONLY)
configure_file(${MEDV4D_CMAKE_SOURCE_DIR}/../bin/qt/QtCored4.dll
    ${MEDV4D_CMAKE_SOURCE_DIR}/../executables/Release/QtCored4.dll COPYONLY)
configure_file(${MEDV4D_CMAKE_SOURCE_DIR}/../bin/qt/QtGui4.dll
    ${MEDV4D_CMAKE_SOURCE_DIR}/../executables/Release/QtGui4.dll COPYONLY)
configure_file(${MEDV4D_CMAKE_SOURCE_DIR}/../bin/qt/QtGuid4.dll
    ${MEDV4D_CMAKE_SOURCE_DIR}/../executables/Release/QtGuid4.dll COPYONLY)
configure_file(${MEDV4D_CMAKE_SOURCE_DIR}/../bin/qt/QtOpenGL4.dll
    ${MEDV4D_CMAKE_SOURCE_DIR}/../executables/Release/QtOpenGL4.dll COPYONLY)
configure_file(${MEDV4D_CMAKE_SOURCE_DIR}/../bin/qt/QtOpenGLd4.dll
    ${MEDV4D_CMAKE_SOURCE_DIR}/../executables/Release/QtOpenGLd4.dll COPYONLY)
