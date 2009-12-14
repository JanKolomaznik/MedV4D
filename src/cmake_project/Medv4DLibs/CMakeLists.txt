CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
PROJECT(MEDV4D)

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
MEDV4D_LIBRARY_TARGET_PREPARATION(vtkIntegration "${MEDV4D_CMAKE_SOURCE_DIR}/../vtkIntegration/src" "${MEDV4D_CMAKE_SOURCE_DIR}/../vtkIntegration")
ADD_DEPENDENCIES(vtkIntegration Common Imaging)
#--------------------------------------------------------
#       GUI
AUX_SOURCE_DIRECTORY(${MEDV4D_CMAKE_SOURCE_DIR}/../GUI/widgets/src WidgetsSrc )
AUX_SOURCE_DIRECTORY(${MEDV4D_CMAKE_SOURCE_DIR}/../GUI/widgets/ogl/src WidgetsOglSrc )
AUX_SOURCE_DIRECTORY(${MEDV4D_CMAKE_SOURCE_DIR}/../GUI/widgets/utils/src WidgetsUtilsSrc )
AUX_SOURCE_DIRECTORY(${MEDV4D_CMAKE_SOURCE_DIR}/../GUI/utils/src UtilsSrc )
AUX_SOURCE_DIRECTORY(${MEDV4D_CMAKE_SOURCE_DIR}/../GUI/widgets/components/src WidgetsComponentsSrc )

FILE( GLOB rccinput ${MEDV4D_CMAKE_SOURCE_DIR}/../GUI/widgets/src/*.qrc )
FILE( GLOB GUI_header_files ${MEDV4D_CMAKE_SOURCE_DIR}/../GUI/utils/*.h ${MEDV4D_CMAKE_SOURCE_DIR}/../GUI/widgets/*.h ${MEDV4D_CMAKE_SOURCE_DIR}/../GUI/widgets/components/*.h ${MEDV4D_CMAKE_SOURCE_DIR}/../GUI/widgets/utils/*.h )
FILTER_HEADERS_FOR_MOC( "${GUI_header_files}" mocinput )

QT4_WRAP_CPP(mocoutput ${mocinput})
QT4_ADD_RESOURCES(rccoutput ${rccinput} ) #OPTIONS --root ../tmp

SOURCE_GROUP( GUI_Sources FILES ${WidgetsSrc} ${WidgetsOglSrc} ${WidgetsUtilsSrc} ${UtilsSrc} ${WidgetsComponentsSrc} )
SOURCE_GROUP( GUI_Resources FILES  ${rccinput} )
SOURCE_GROUP( GUI_Generated FILES  ${mocoutput} ${rccoutput} )
SOURCE_GROUP( GUI_Headers FILES ${GUI_header_files} )

ADD_LIBRARY(m4dWidgets ${WidgetsSrc} ${WidgetsOglSrc} ${WidgetsUtilsSrc} ${UtilsSrc} ${WidgetsComponentsSrc} ${mocoutput} ${rccoutput} ${GUI_header_files} ${rccinput} )
ADD_DEPENDENCIES(m4dWidgets Common Imaging backendForDICOM vtkIntegration)

#--------------------------------------------------------
# Set for parent CMakeLists
SET( MEDV4D_LIBS_CMAKE_PROCESSED TRUE PARENT_SCOPE)
