#vtkIntegration library

include_directories(../../ ../../include ${VTK_INCLUDE_DIRS})
link_directories(../lib )

set(SRC_DIR ../../vtkIntegration/src)

set(INPUT ${SRC_DIR}/DataConversion.cpp 
          ${SRC_DIR}/m4dImageDataSource.cpp)

add_library(vtkIntegration ${INPUT})

add_dependencies(vtkIntegration Common)

