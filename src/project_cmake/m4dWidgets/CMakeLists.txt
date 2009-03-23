#m4dWidgets library

include(${QT_USE_FILE})
include_directories(../../ ../../include 
                    ${QT_INCLUDE_DIR}
                    ${QT_QT_INCLUDE_DIR}
                    ${QT_QTCORE_INCLUDE_DIR}
                    ${QT_QTGUI_INCLUDE_DIR}
                    ${QT_QTOPENGL_INCLUDE_DIR}
                    ${VTK_INCLUDE_DIRS})

set(GUI_INCLUDE ../../GUI/widgets)

set(MOCINPUT ${GUI_INCLUDE}/m4dGUIAbstractViewerWidget.h
             ${GUI_INCLUDE}/m4dGUIMainWindow.h 
             ${GUI_INCLUDE}/m4dGUIMainWindow2.h
             ${GUI_INCLUDE}/m4dGUIMainViewerDesktopWidget.h
             ${GUI_INCLUDE}/m4dGUIScreenLayoutWidget.h
             ${GUI_INCLUDE}/m4dGUIStudyManagerWidget.h 
             ${GUI_INCLUDE}/m4dGUIVtkViewerWidget.h
             ${GUI_INCLUDE}/m4dGUISliceViewerWidget.h
             ${GUI_INCLUDE}/m4dGUISliceViewerWidget2.h
             ${GUI_INCLUDE}/m4dGUIToolBarCustomizerWidget.h
             ${GUI_INCLUDE}/m4dGUIProgressBarWidget.h 
             ${GUI_INCLUDE}/components/StManagerFilterComp.h
             ${GUI_INCLUDE}/components/StManagerHProtocolsComp.h
             ${GUI_INCLUDE}/components/StManagerStudyListComp.h
             ${GUI_INCLUDE}/utils/LoadingThreads.h
             ${GUI_INCLUDE}/utils/StudyFilter.h)

QT4_WRAP_CPP(MOCOUTPUT ${MOCINPUT})

set(SRC_DIR ../../GUI/widgets)

set(RCCINPUT ${SRC_DIR}/src/m4dGUIMainWindow.qrc 
             ${SRC_DIR}/src/m4dGUIMainWindow2.qrc
             ${SRC_DIR}/src/m4dGUIStudyManagerWidget.qrc
             ${SRC_DIR}/src/m4dGUIScreenLayoutWidget.qrc
             ${SRC_DIR}/src/m4dGUIToolBarCustomizerWidget.qrc)

QT4_ADD_RESOURCES(RCCOUTPUT ${RCCINPUT})

set(INPUT ${SRC_DIR}/src/m4dGUIMainWindow.cpp 
          ${SRC_DIR}/src/m4dGUIMainWindow2.cpp
          ${SRC_DIR}/src/m4dGUIMainViewerDesktopWidget.cpp
          ${SRC_DIR}/src/m4dGUIVtkViewerWidget.cpp 
          ${SRC_DIR}/src/m4dGUIStudyManagerWidget.cpp
          ${SRC_DIR}/src/m4dGUIScreenLayoutWidget.cpp 
          ${SRC_DIR}/src/m4dGUISliceViewerWidget.cpp
          ${SRC_DIR}/src/m4dGUISliceViewerWidget2.cpp
          ${SRC_DIR}/src/m4dGUIToolBarCustomizerWidget.cpp
          ${SRC_DIR}/src/m4dGUIProgressBarWidget.cpp 
          ${SRC_DIR}/ogl/src/fonts.cpp
          ${SRC_DIR}/components/src/StManagerFilterComp.cpp
          ${SRC_DIR}/components/src/StManagerHProtocolsComp.cpp
          ${SRC_DIR}/components/src/StManagerStudyListComp.cpp 
          ${SRC_DIR}/utils/src/LoadingThreads.cpp
          ${SRC_DIR}/utils/src/StudyFilter.cpp)

add_library(m4dWidgets ${INPUT} ${MOCOUTPUT} ${RCCOUTPUT})

add_dependencies(m4dWidgets Common)

