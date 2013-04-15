INCLUDE(FindPackageHandleStandardArgs)

SET( SOGLU_ROOT_DIR "SOGLU_ROOT_DIR-NOTFOUND" CACHE PATH "Path to SOGLU_ROOT_DIR" )

FIND_PATH(SOGLU_INCLUDE_DIR CgFXShader.hpp 
	HINTS ${SOGLU_ROOT_DIR} ${SOGLU_ROOT_DIR}/include
	PATH_SUFFIXES soglu
	DOC "The path the the directory that contains SOGLU.h"
)

FIND_LIBRARY(SOGLU_LIBRARIES_RELEASE
	PATHS ${SOGLU_ROOT_DIR}
	NAMES soglu
	PATH_SUFFIXES lib 
	DOC "libSOGLU_x86_64"
)

FIND_LIBRARY(SOGLU_LIBRARIES_DEBUG
	PATHS ${SOGLU_ROOT_DIR}
	NAMES soglu_d
	PATH_SUFFIXES lib
	DOC "libSOGLU_x86_64"
)

SET( SOGLU_LIBRARIES debug ${SOGLU_LIBRARIES_DEBUG} optimized ${SOGLU_LIBRARIES_RELEASE} ) 

FIND_PACKAGE_HANDLE_STANDARD_ARGS(SOGLU DEFAULT_MSG SOGLU_LIBRARIES SOGLU_INCLUDE_DIR SOGLU_ROOT_DIR)
