include(${QT_USE_FILE})
include_directories(../../ ../../include /usr/local/include ${QT_INCLUDE_DIR} ${QT_QT_INCLUDE_DIR} ${QT_QTCORE_INCLUDE_DIR} ${QT_QTGUI_INCLUDE_DIR} ${QT_QTOPENGL_INCLUDE_DIR} ${VTK_INCLUDE_DIRS})

set(GUI_INCLUDE ../../include/GUI)

set(MOCINPUT ${GUI_INCLUDE}/m4dGUIAbstractViewerWidget.h ${GUI_INCLUDE}/m4dGUIMainWindow.h ${GUI_INCLUDE}/m4dGUIMainWindow2.h ${GUI_INCLUDE}/m4dGUIMainViewerDesktopWidget.h ${GUI_INCLUDE}/m4dGUIScreenLayoutWidget.h ${GUI_INCLUDE}/m4dGUIStudyManagerWidget.h ${GUI_INCLUDE}/m4dGUIVtkViewerWidget.h ${GUI_INCLUDE}/m4dGUISliceViewerWidget.h ${GUI_INCLUDE}/m4dGUISliceViewerWidget2.h ${GUI_INCLUDE}/m4dGUIToolBarCustomizerWidget.h ${GUI_INCLUDE}/m4dGUIProgressBarWidget.h ${GUI_INCLUDE}/StManagerFilterComp.h ${GUI_INCLUDE}/StManagerHProtocolsComp.h ${GUI_INCLUDE}/StManagerStudyListComp.h)

QT4_WRAP_CPP(MOCOUTPUT ${MOCINPUT})

set(SRC_DIR ../../GUI/widgets)

set(RCCINPUT ${SRC_DIR}/m4dGUIMainWindow.qrc ${SRC_DIR}/m4dGUIMainWindow2.qrc ${SRC_DIR}/m4dGUIStudyManagerWidget.qrc ${SRC_DIR}/m4dGUIScreenLayoutWidget.qrc ${SRC_DIR}/m4dGUIToolBarCustomizerWidget.qrc)

QT4_ADD_RESOURCES(RCCOUTPUT ${RCCINPUT})

set(INPUT ${SRC_DIR}/m4dGUIMainWindow.cpp ${SRC_DIR}/m4dGUIMainWindow2.cpp ${SRC_DIR}/m4dGUIMainViewerDesktopWidget.cpp ${SRC_DIR}/m4dGUIVtkViewerWidget.cpp ${SRC_DIR}/m4dGUIStudyManagerWidget.cpp ${SRC_DIR}/m4dGUIScreenLayoutWidget.cpp ${SRC_DIR}/m4dGUISliceViewerWidget.cpp ${SRC_DIR}/m4dGUISliceViewerWidget2.cpp ${SRC_DIR}/m4dGUIToolBarCustomizerWidget.cpp ${SRC_DIR}/m4dGUIProgressBarWidget.cpp ${SRC_DIR}/ogl/fonts.cpp ${SRC_DIR}/components/StManagerFilterComp.cpp ${SRC_DIR}/components/StManagerHProtocolsComp.cpp ${SRC_DIR}/components/StManagerStudyListComp.cpp ${SRC_DIR}/components/StudyFilter.cpp)

add_library(m4dWidgets ${INPUT} ${MOCOUTPUT} ${RCCOUTPUT})
