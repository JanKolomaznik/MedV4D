#VolumeDataFusions application

include(${QT_USE_FILE})

include_directories(../../ ../../include 
                    ${QT_INCLUDE_DIR} 
                    ${QT_QT_INCLUDE_DIR} 
                    ${QT_QTCORE_INCLUDE_DIR} 
                    ${QT_QTGUI_INCLUDE_DIR} 
                    ${QT_QTOPENGL_INCLUDE_DIR} 
                    ${VTK_INCLUDE_DIRS})

link_directories(${VTK_LIBRARY_DIRS} 
                 ${DCMTK_LIBRARY_DIRS} 
                 ${Boost_LIBRARY_DIRS} 
                 ${QBoost_LIBRARY_DIRS})

set(VTK_LIBRARIES vtkCommon vtkDICOMParser vtkexoIIc vtkFiltering vtkftgl vtkGenericFiltering vtkGraphics vtkHybrid vtkImaging vtkIO vtkNetCDF vtksys vtkVolumeRendering vtkWidgets QVTK vtkRendering)

IF(UNIX)
  set(VTK_LIBRARIES ${VTK_LIBRARIES} wrap)
ENDIF(UNIX)

set(DCMTK_LIBRARIES dcmnet dcmdata ofstd)

set(M4DLIBS Imaging m4dWidgets vtkIntegration Common backendForDICOM)



set(SRC_DIR ../../Applications/VolumeDataFusions)

set(MOCINPUT ${SRC_DIR}/mainWindow.h
	     ${SRC_DIR}/SettingsBox.h)

QT4_WRAP_CPP(MOCOUTPUT ${MOCINPUT})

set(RCCINPUT ${SRC_DIR}/mainWindow.qrc)

QT4_ADD_RESOURCES(RCCOUTPUT ${RCCINPUT})

set(INPUT ${SRC_DIR}/main.cpp
	  ${SRC_DIR}/mainWindow.cpp
	  ${SRC_DIR}/SettingsBox.cpp)



set(OUTPUT VolumeDataFusions)

add_executable(${OUTPUT} ${INPUT} ${MOCOUTPUT} ${RCCOUTPUT})
target_link_libraries(${OUTPUT} 
                      ${M4DLIBS} 
                      ${Boost_LIBRARIES} 
                      ${VTK_LIBRARIES} 
                      ${QT_LIBRARIES} 
                      ${OPENGL_LIBRARY} 
                      ${DCMTK_LIBRARIES})

add_dependencies(${OUTPUT} Imaging m4dWidgets vtkIntegration Common backendForDICOM)

