
IF(WIN32)
	SET(CMAKE_CXX_FLAGS_DEBUG "/MTd /ZI /Od -DDEBUG_LEVEL=10 -DDEBUG_ADITIONAL_INFO")
	SET(CMAKE_CXX_FLAGS_RELEASE "/MT")
ELSE(WIN32)
	SET(CMAKE_CXX_FLAGS "-Wall -Wno-deprecated -DDCOMPAT_H -DHAVE_CONFIG_H -DHAVE_SSTREAM" )
	SET(CMAKE_CXX_FLAGS_DEBUG "-ggdb -DDEBUG_LEVEL=10 -DDEBUG_ADITIONAL_INFO")
	SET(CMAKE_CXX_FLAGS_RELEASE "-O3")
ENDIF(WIN32)
