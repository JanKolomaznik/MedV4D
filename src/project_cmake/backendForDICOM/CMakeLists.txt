include_directories(../../backendForDICOM ../../include ../..)

set(SRC_DIR ../../backendForDICOM/src)
set(INPUT ${SRC_DIR}/AbstractService.cpp ${SRC_DIR}/DicomAssoc.cpp ${SRC_DIR}/DicomObj.cpp ${SRC_DIR}/DICOMServiceProvider.cpp ${SRC_DIR}/FindService.cpp ${SRC_DIR}/LocalService.cpp ${SRC_DIR}/MoveService.cpp ${SRC_DIR}/StoreService.cpp)

add_library(backendForDICOM ${INPUT})
