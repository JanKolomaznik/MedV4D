#
# nešlo vygenerovať
#
CMAKE_MINIMUM_REQUIRED( VERSION 2.8 )

SET( CMAKE_CXX_COMPILER g++ )
SET( CMAKE_C_COMPILER gcc )

PROJECT ( Hmira )

IF( EXTENSION_MODULES_ENABLED )
	SET( EXTENSION_HEADERS
		"concepts/iterable_vertices.h"
		"concepts/mesh.h"
		"concepts/vertex_adjacency.h"
		)
ENDIF( EXTENSION_MODULES_ENABLED )

SET( HMIRA_HEADERS 
	"OpenMeshX.h"
	"winged_edge.h" )


INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR} )

LINK_LIBRARIES(
	${Boost_LIBRARIES} 
	OpenMeshCored)

ADD_EXECUTABLE( main "main.cpp" ${HMIRA_HEADERS} ${EXTENSION_HEADERS} )

