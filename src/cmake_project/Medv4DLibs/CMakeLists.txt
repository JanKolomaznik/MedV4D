CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
PROJECT(MEDV4D)

if(POLICY CMP0012)
	cmake_policy(SET CMP0012 NEW)
endif(POLICY CMP0012)
if(POLICY CMP0011)
	cmake_policy(SET CMP0011 NEW)
endif(POLICY CMP0011)


SET(VTK_ENABLED true CACHE BOOL "Enable components with VTK dependencies." )
SET(CG_ENABLED true CACHE BOOL "Enable components with Cg shader dependencies." )
SET(OLD_M4D_WIDGETS_ENABLED true CACHE BOOL "Enable if you want to use old set of widgets" )


IF(NOT MEDV4D_CMAKE_SOURCE_DIR)
	SET(MEDV4D_CMAKE_SOURCE_DIR "${CMAKE_SOURCE_DIR}/.." )
	SET(MEDV4D_CMAKE_MODULES_DIR ${MEDV4D_CMAKE_SOURCE_DIR}/Modules)
	SET(CMAKE_MODULE_PATH "${MEDV4D_CMAKE_MODULES_DIR}" "${CMAKE_ROOT}/Modules" )
ENDIF(NOT MEDV4D_CMAKE_SOURCE_DIR)

SET(LIBRARY_OUTPUT_PATH ${MEDV4D_CMAKE_SOURCE_DIR}/../lib)
SET(EXECUTABLE_OUTPUT_PATH ${MEDV4D_CMAKE_SOURCE_DIR}/../executables)
SET(CMAKE_DEBUG_POSTFIX, "d") 

INCLUDE( "${MEDV4D_CMAKE_MODULES_DIR}/Functions.cmake" )
INCLUDE( "${MEDV4D_CMAKE_MODULES_DIR}/FindPackages.cmake" )
INCLUDE( "${MEDV4D_CMAKE_MODULES_DIR}/StandardCompileOptions.cmake" ) 

IF( ${VTK_ENABLED} MATCHES "ON" )
	SET( EXP_COMPILE_DEFINITIONS ${EXP_COMPILE_DEFINITIONS} "-DUSE_VTK" )
	message( "VTK aaaaaaaaaaa" )
ENDIF( ${VTK_ENABLED} MATCHES "ON" )
IF( ${CG_ENABLED} MATCHES "ON" )
	SET( EXP_COMPILE_DEFINITIONS ${EXP_COMPILE_DEFINITIONS} "-DUSE_CG" )
ENDIF( ${CG_ENABLED} MATCHES "ON" )

ADD_DEFINITIONS( ${EXP_COMPILE_DEFINITIONS} )

#--------------------------------------------------------
#       COMMON
MEDV4D_LIBRARY_TARGET_PREPARATION(Common "${MEDV4D_CMAKE_SOURCE_DIR}/../common/src" "${MEDV4D_CMAKE_SOURCE_DIR}/../common")
#--------------------------------------------------------
#       Imaging
MEDV4D_LIBRARY_TARGET_PREPARATION(Imaging "${MEDV4D_CMAKE_SOURCE_DIR}/../Imaging/src" "${MEDV4D_CMAKE_SOURCE_DIR}/../Imaging")
ADD_DEPENDENCIES(Imaging Common)
#--------------------------------------------------------
#       backendForDICOM
MEDV4D_LIBRARY_TARGET_PREPARATION(backendForDICOM "${MEDV4D_CMAKE_SOURCE_DIR}/../backendForDICOM/src" "${MEDV4D_CMAKE_SOURCE_DIR}/../backendForDICOM")
ADD_DEPENDENCIES(backendForDICOM Common Imaging )

IF( DCMTK_OPTIONS )
	SET_TARGET_PROPERTIES( backendForDICOM PROPERTIES COMPILE_DEFINITIONS ${DCMTK_OPTIONS} )
ENDIF( DCMTK_OPTIONS )


#--------------------------------------------------------
#       vtkIntegration
SET(VTK_INTEGRATION_TARGET "")
IF( ${VTK_ENABLED} MATCHES "ON" )
	MEDV4D_LIBRARY_TARGET_PREPARATION(vtkIntegration "${MEDV4D_CMAKE_SOURCE_DIR}/../vtkIntegration/src" "${MEDV4D_CMAKE_SOURCE_DIR}/../vtkIntegration")
	ADD_DEPENDENCIES(vtkIntegration Common Imaging)
	SET(VTK_INTEGRATION_TARGET vtkIntegration)
ENDIF( ${VTK_ENABLED} MATCHES "ON" )
#--------------------------------------------------------
#       GUI


AUX_SOURCE_DIRECTORY(${MEDV4D_CMAKE_SOURCE_DIR}/../GUI/widgets/src WidgetsSrc )
AUX_SOURCE_DIRECTORY(${MEDV4D_CMAKE_SOURCE_DIR}/../GUI/widgets/ogl/src WidgetsOglSrc )
AUX_SOURCE_DIRECTORY(${MEDV4D_CMAKE_SOURCE_DIR}/../GUI/widgets/utils/src WidgetsUtilsSrc )
AUX_SOURCE_DIRECTORY(${MEDV4D_CMAKE_SOURCE_DIR}/../GUI/utils/src UtilsSrc )

IF( ${OLD_M4D_WIDGETS_ENABLED} MATCHES "ON" )
	AUX_SOURCE_DIRECTORY(${MEDV4D_CMAKE_SOURCE_DIR}/../GUI/widgets/components/src WidgetsComponentsSrc )
ENDIF( ${OLD_M4D_WIDGETS_ENABLED} MATCHES "ON" )

FILE( GLOB rccinput ${MEDV4D_CMAKE_SOURCE_DIR}/../GUI/widgets/src/*.qrc )
FILE( GLOB uiinput ${MEDV4D_CMAKE_SOURCE_DIR}/../GUI/widgets/*.ui )
FILE( GLOB GUI_header_files ${MEDV4D_CMAKE_SOURCE_DIR}/../GUI/utils/*.h ${MEDV4D_CMAKE_SOURCE_DIR}/../GUI/widgets/*.h ${MEDV4D_CMAKE_SOURCE_DIR}/../GUI/widgets/components/*.h ${MEDV4D_CMAKE_SOURCE_DIR}/../GUI/widgets/utils/*.h )

IF( ${OLD_M4D_WIDGETS_ENABLED} MATCHES "OFF"  )
	FILTER_FILES_FOR_STRING( "/m4d" "${WidgetsSrc}" tmpList1 tmpList2 )
	SET( WidgetsSrc ${tmpList2} )
	FILTER_FILES_FOR_STRING( "/m4d" "${rccinput}" tmpList1 tmpList2 )
	SET( rccinput ${tmpList2} )
	FILTER_FILES_FOR_STRING( "/m4d" "${GUI_header_files}" tmpList1 tmpList2 )
	SET( GUI_header_files ${tmpList2} )
ENDIF( ${OLD_M4D_WIDGETS_ENABLED} MATCHES "OFF" )

FILTER_HEADERS_FOR_MOC( "${GUI_header_files}" mocinput )

QT4_WRAP_CPP(mocoutput ${mocinput} OPTIONS ${EXP_COMPILE_DEFINITIONS} )
QT4_ADD_RESOURCES(rccoutput ${rccinput} ) #OPTIONS --root ../tmp
QT4_WRAP_UI(uioutput ${uiinput} )


SOURCE_GROUP( GUI_Sources FILES ${WidgetsSrc} ${WidgetsOglSrc} ${WidgetsUtilsSrc} ${UtilsSrc} ${WidgetsComponentsSrc} )
SOURCE_GROUP( GUI_Resources FILES  ${rccinput} )
SOURCE_GROUP( GUI_Generated FILES  ${mocoutput} ${rccoutput} ${uioutput} )
SOURCE_GROUP( GUI_Headers FILES ${GUI_header_files} )

ADD_LIBRARY(m4dWidgets ${WidgetsSrc} ${WidgetsOglSrc} ${WidgetsUtilsSrc} ${UtilsSrc} ${WidgetsComponentsSrc} ${mocoutput} ${rccoutput} ${uioutput} ${GUI_header_files} ${rccinput} )
ADD_DEPENDENCIES(m4dWidgets Common Imaging backendForDICOM ${VTK_INTEGRATION_TARGET} )

#--------------------------------------------------------
# Set for parent CMakeLists
SET( MEDV4D_LIBS_CMAKE_PROCESSED TRUE PARENT_SCOPE)
SET( MEDV4D_COMPILE_DEFINITIONS ${EXP_COMPILE_DEFINITIONS} PARENT_SCOPE )
