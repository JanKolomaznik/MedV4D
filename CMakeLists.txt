CMAKE_MINIMUM_REQUIRED(VERSION 2.8.9)
PROJECT( M4DViewer )

if(WIN32)
	#find_path(QT_ROOT_DIRECTORY "bin/designer.exe" PATHS "C:/Qt/Qt5.3.0/5.3/msvc2013_64_opengl")
	find_path(WINDOWS_KIT_DIRECTORY "User32.Lib" PATHS "c:/Program Files (x86)/Windows Kits/8.1/Lib/winv6.3/um/x64/")
	set(CMAKE_PREFIX_PATH ${QT_ROOT_DIRECTORY} ${WINDOWS_KIT_DIRECTORY} ${CMAKE_PREFIX_PATH})
	# preprocessor macros generate warning c4003 while accepting empty __VA_ARGS__ in MSVC 2012
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4003 /bigobj")

	#On windows
	#SET(CMAKE_PREFIX_PATH "C:/Libs/Qt/Qt5.2.1/5.2.1/msvc2012_64_opengl" "c:/Program Files (x86)/Windows Kits/8.0/Lib/win8/um/x64/")

endif(WIN32)

IF(NOT MEDV4D_ROOT_DIRECTORY)
	SET(MEDV4D_ROOT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/MedV4D" )
ENDIF(NOT MEDV4D_ROOT_DIRECTORY)

IF(NOT MEDV4D_CMAKE_MODULES_DIR )
	SET(MEDV4D_CMAKE_MODULES_DIR ${MEDV4D_ROOT_DIRECTORY}/../cmake_project/Modules)
	SET(CMAKE_MODULE_PATH "${MEDV4D_CMAKE_MODULES_DIR}" "${CMAKE_ROOT}/Modules" )
ENDIF(NOT MEDV4D_CMAKE_MODULES_DIR )


SET( MEDV4D_ROOT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/MedV4D )
SET( MEDV4D_LIB_OUTPUT_DIRECTORY ${MEDV4D_ROOT_DIRECTORY}/lib )
SET( MEDV4D_EXECUTABLES_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/executables )


INCLUDE( "${MEDV4D_CMAKE_MODULES_DIR}/SetupOutputPrefixAndPostfix.cmake" )

include_directories(
	"../../extern/tfw"
	"../../extern/soglu"
	"../../extern/vorgl"
	"../../extern/cugip"
	"../../extern/prognot"
	)

ADD_SUBDIRECTORY( MedV4D )
ADD_SUBDIRECTORY( Applications/Viewer )
#ADD_SUBDIRECTORY( Applications/tools )
ADD_DEPENDENCIES( Viewer MedV4D )
#ADD_DEPENDENCIES( CudaFilters MedV4D )
