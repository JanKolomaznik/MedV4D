set(SRC_DIR ../../cell/cellCommon)
set(INPUT ${SRC_DIR}/BasicJob.cpp ${SRC_DIR}/FilterSerializerArray.cpp ${SRC_DIR}/GeneralDataSetSerializer.cpp ${SRC_DIR}/GeneralFilterSerializer.cpp)

add_library(cellCommon ${INPUT})
