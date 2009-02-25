set(SRC_DIR ../../cell/cellClient)
set(INPUT ${SRC_DIR}/cellClient.cpp ${SRC_DIR}/clientJob.cpp ${SRC_DIR}/clientSocket.cpp ${SRC_DIR}/remoteFilterBase.cpp)

add_library(cellClient ${INPUT})
