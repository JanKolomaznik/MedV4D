set(SRC_DIR ../../cell/cellServer)
set(INPUT ${SRC_DIR}/server.cpp ${SRC_DIR}/main.cpp ${SRC_DIR}/serverJob.cpp)

set(OUTPUT cellServer)

set(M4DLIBS cellCommon Imaging backendForDICOM Common)

add_executable(${OUTPUT} ${INPUT})
target_link_libraries(${OUTPUT} ${M4DLIBS} ${Boost_LIBRARIES})
