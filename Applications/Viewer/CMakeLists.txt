CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
CMAKE_POLICY(VERSION 2.8)
PROJECT( Viewer )

IF(NOT MEDV4D_ROOT_DIRECTORY)
	SET(MEDV4D_ROOT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../../MedV4D" )
ENDIF(NOT MEDV4D_ROOT_DIRECTORY)

IF(NOT MEDV4D_CMAKE_MODULES_DIR )
	SET(MEDV4D_CMAKE_MODULES_DIR ${MEDV4D_ROOT_DIRECTORY}/../cmake_project/Modules)
	SET(CMAKE_MODULE_PATH "${MEDV4D_CMAKE_MODULES_DIR}" "${CMAKE_ROOT}/Modules" )
ENDIF(NOT MEDV4D_CMAKE_MODULES_DIR )

FIND_PACKAGE( MedV4D REQUIRED )

###############################
## OIS
OPTION(OIS_ENABLED "Enable components with OIS support" OFF)
IF(OIS_ENABLED)
	MESSAGE(STATUS "OIS support enabled")
	ADD_DEFINITIONS(-DOIS_ENABLED)

	FIND_PACKAGE(OIS REQUIRED)

	INCLUDE_DIRECTORIES( ${OIS_INCLUDE_DIR} )
ELSE(OIS_ENABLED)
	MESSAGE(STATUS "OIS support disabled")
ENDIF(OIS_ENABLED)
## OIS
###############################

SET(EXECUTABLE_OUTPUT_PATH ${MEDV4D_EXECUTABLES_OUTPUT_DIRECTORY} )

OPTION(EXTENSION_MODULES_ENABLED "Enable applications extension modules" OFF)

IF( EXTENSION_MODULES_ENABLED )
	ADD_DEFINITIONS(-DEXTENSION_MODULES_ENABLED)
	SET( EXTENSION_SOURCES 
		"AnnotationModule/AnnotationEditorController.cpp"
		"ShoulderMeasurementModule/Computations.cpp"
		"ShoulderMeasurementModule/ShoulderMeasurementController.cpp"
		"ShoulderMeasurementModule/ShoulderMeasurementModule.cpp"
		"OrganSegmentationModule/OrganSegmentationModule.cpp"
		"OrganSegmentationModule/OrganSegmentationController.cpp"
		"OrganSegmentationModule/OrganSegmentationWidget.cpp"
	)
	SET( EXTENSION_HEADERS
		"AnnotationModule/AnnotationEditorController.hpp"
		"AnnotationModule/AnnotationModule.hpp"
		"AnnotationModule/AnnotationSettingsDialog.hpp"
		"AnnotationModule/AnnotationWidget.hpp"
		"ShoulderMeasurementModule/Computations.hpp"
		"ShoulderMeasurementModule/ShoulderMeasurementModule.hpp"
		"ShoulderMeasurementModule/ShoulderMeasurementWidget.hpp"
		"ShoulderMeasurementModule/ShoulderMeasurementController.hpp"
		"OrganSegmentationModule/OrganSegmentationModule.hpp"
		"OrganSegmentationModule/OrganSegmentationWidget.hpp"
		"OrganSegmentationModule/OrganSegmentationController.hpp"
		)
	SET( EXTENSION_UIS
		"AnnotationModule/AnnotationSettingsDialog.ui"
		"AnnotationModule/AnnotationWidget.ui"
		"ShoulderMeasurementModule/ShoulderMeasurementWidget.ui"
		"OrganSegmentationModule/OrganSegmentationWidget.ui"
		)
ENDIF( EXTENSION_MODULES_ENABLED )


SET( VIEWER_HEADERS 
	"JoyInput.h"
	"ViewerWindow.hpp" )
SET( VIEWER_SOURCES 
	"JoyInput.cpp"
	"main.cpp"
	"ViewerWindow.cpp" )
SET( VIEWER_UI
	"ViewerWindow.ui"
	)

LINK_DIRECTORIES( ${MEDV4D_LIB_DIRS_3RD_PARTY} )
INCLUDE_DIRECTORIES( ${MEDV4D_INCLUDE_DIR} ${MEDV4D_INCLUDE_DIRS_3RD_PARTY} ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR} )

INCLUDE( "${MEDV4D_CMAKE_MODULES_DIR}/StandardCompileOptions.cmake" NO_POLICY_SCOPE)
ADD_MEDV4D_EXECUTABLE( Viewer SOURCES ${VIEWER_SOURCES} ${EXTENSION_SOURCES} HEADERS ${VIEWER_HEADERS} ${EXTENSION_HEADERS} UIS ${VIEWER_UI} ${EXTENSION_UIS} )
TARGET_LINK_LIBRARIES( Viewer ${MEDV4D_LIBRARIES} ${MEDV4D_LIBRARIES_3RD_PARTY} )
###############################
## OIS
IF(OIS_ENABLED)
	TARGET_LINK_LIBRARIES(Viewer ${OIS_LIBRARIES})
ENDIF(OIS_ENABLED)
## OIS
###############################
