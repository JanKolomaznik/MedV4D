#common library

include_directories(../../common ../../include)

set(SRC_DIR ../../common/src)
set(INPUT ${SRC_DIR}/Common.cpp 
          ${SRC_DIR}/Debug.cpp 
          ${SRC_DIR}/ExceptionBase.cpp 
          ${SRC_DIR}/Functors.cpp 
          ${SRC_DIR}/Log.cpp 
          ${SRC_DIR}/TimeStamp.cpp 
          ${SRC_DIR}/IOStreams.cpp)

add_library(Common ${INPUT})
