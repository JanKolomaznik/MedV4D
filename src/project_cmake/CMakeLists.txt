#MedV4D Project "solution"

CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
PROJECT(MEDV4D)

SET(LIBRARY_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/../lib)
SET(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/../executables)
SET(CMAKE_DEBUG_POSTFIX, "d") 

SET(QT_USE_QTOPENGL 1)

FIND_PACKAGE(Boost REQUIRED COMPONENTS filesystem system thread)
FIND_PACKAGE(OpenGL REQUIRED)

IF(WIN32)
  FIND_PACKAGE(Qt4 REQUIRED)
  get_filename_component(QT_MOC_EXECUTABLE ../bin/qt/moc.exe ABSOLUTE)
  get_filename_component(QT_UIC_EXECUTABLE ../bin/qt/uic.exe ABSOLUTE)
  get_filename_component(QT_RCC_EXECUTABLE ../bin/qt/rcc.exe ABSOLUTE)
  SET(QT_INCLUDE_DIR "../../include/qt/headers")
  SET(QT_QT_INCLUDE_DIR "../../include/qt/headers")
  SET(QT_QTCORE_INCLUDE_DIR "../../include/qt/headers/QtOpenGL")
  SET(QT_QTGUI_INCLUDE_DIR "../../include/qt/headers/QtGui")
  SET(QT_QTOPENGL_INCLUDE_DIR "../../include/qt/headers/QtCore")
  SET(QT_LIBRARIES ../../lib/qt/QtOpenGLd.lib 
                   ../../lib/qt/QtGuid.lib
                   ../../lib/qt/QtCored.lib)

  SET(VTK_LIBRARY_DIRS "../../lib/vtk")
  SET(VTK_INCLUDE_DIRS "../../include/vtk")

  SET(DCMTK_LIBRARY_DIRS "../../lib/dcmtkLibs")
  SET(DCMTK_INCLUDE_DIRS "../../include/dcmtk")
ELSE(WIN32)
  FIND_PACKAGE(Qt4 REQUIRED)
  FIND_PACKAGE(VTK REQUIRED)
  FIND_PACKAGE(DCMTK REQUIRED)
ENDIF(WIN32)

IF(WIN32)
  SET(CMAKE_CXX_FLAGS "/MT /EHsc -DDEBUG_LEVEL=10 -DDEBUG_ADITIONAL_INFO")
ELSE(WIN32)
  SET(CMAKE_CXX_FLAGS "-ggdb -Wall -Wno-deprecated -DDCOMPAT_H -DHAVE_CONFIG_H -DHAVE_SSTREAM -DDEBUG_LEVEL=10 -DDEBUG_ADITIONAL_INFO")
ENDIF(WIN32)

ADD_SUBDIRECTORY(common)
ADD_SUBDIRECTORY(backendForDICOM)
ADD_SUBDIRECTORY(Imaging)
ADD_SUBDIRECTORY(m4dWidgets)
ADD_SUBDIRECTORY(vtkIntegration)

ADD_SUBDIRECTORY(m4dPilot)

