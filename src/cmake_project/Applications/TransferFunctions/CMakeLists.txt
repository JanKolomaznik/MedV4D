CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
PROJECT( TransferFunctions )

SET(MEDV4D_CMAKE_SOURCE_DIR "${CMAKE_SOURCE_DIR}/../.." )
SET(MEDV4D_CMAKE_MODULES_DIR ${MEDV4D_CMAKE_SOURCE_DIR}/Modules)

INCLUDE( "${MEDV4D_CMAKE_MODULES_DIR}/ApplicationCMakeListMandatoryInclude.cmake" )


SET(EXECUTABLE_OUTPUT_PATH ${MEDV4D_CMAKE_SOURCE_DIR}/../executables)
TARGET_MEDV4D_PROGRAM( TransferFunctions ${MEDV4D_CMAKE_SOURCE_DIR}/../Applications/TransferFunctions )

#copy shader to project dir
configure_file(${MEDV4D_CMAKE_SOURCE_DIR}/../GUI/shaders/ImageRender.cgfx
	${CMAKE_CURRENT_BINARY_DIR}/ImageRender.cgfx COPYONLY)
