include(${QT_USE_FILE})
include_directories(../../ ../../include /usr/local/include ${QT_INCLUDE_DIR} ${QT_QT_INCLUDE_DIR} ${QT_QTCORE_INCLUDE_DIR} ${QT_QTGUI_INCLUDE_DIR} ${QT_QTOPENGL_INCLUDE_DIR} ${VTK_INCLUDE_DIRS})

set(VTK_LIBRARIES vtkCommon vtkDICOMParser vtkexoIIc vtkFiltering vtkftgl vtkGenericFiltering vtkGraphics vtkHybrid vtkImaging vtkIO vtkNetCDF vtksys vtkVolumeRendering vtkWidgets QVTK wrap vtkRendering)

set(DCMTK_LIBRARIES dcmnet dcmdata ofstd)

set(SRC_DIR ../../Applications/SimpleMIP)

set(MOCINPUT ${SRC_DIR}/mainWindow.h ${SRC_DIR}/SettingsBox.h)

QT4_WRAP_CPP(MOCOUTPUT ${MOCINPUT})

set(RCCINPUT ${SRC_DIR}/mainWindow.qrc)

QT4_ADD_RESOURCES(RCCOUTPUT ${RCCINPUT})

set(INPUT ${SRC_DIR}/main.cpp ${SRC_DIR}/mainWindow.cpp ${SRC_DIR}/SettingsBox.cpp)

set(OUTPUT SimpleMIP)

set(M4DLIBS backendForDICOM Common Imaging m4dWidgets vtkIntegration)

add_executable(${OUTPUT} ${INPUT} ${MOCOUTPUT} ${RCCOUTPUT})
target_link_libraries(${OUTPUT} ${M4DLIBS} ${Boost_LIBRARIES} ${VTK_LIBRARIES} ${QT_LIBRARIES} ${OPENGL_LIBRARY} ${DCMTK_LIBRARIES})
