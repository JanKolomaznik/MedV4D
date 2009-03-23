#imaging library

include_directories(../../include ../..)

set(SRC_DIR ../../Imaging/src)
set(INPUT ${SRC_DIR}/AbstractDataSet.cpp 
          ${SRC_DIR}/AbstractFilter.cpp 
          ${SRC_DIR}/AbstractImageData.cpp 
          ${SRC_DIR}/AbstractImageFilter.cpp 
          ${SRC_DIR}/ConnectionInterface.cpp 
          ${SRC_DIR}/Image.cpp 
          ${SRC_DIR}/ImageFactory.cpp 
          ${SRC_DIR}/ModificationManager.cpp 
          ${SRC_DIR}/PipelineContainer.cpp 
          ${SRC_DIR}/Pipeline.cpp 
          ${SRC_DIR}/Ports.cpp 
          ${SRC_DIR}/DataSetFactory.cpp)

add_library(Imaging ${INPUT})

add_dependencies(Imaging Common)

