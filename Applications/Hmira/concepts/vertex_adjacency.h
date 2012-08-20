/*
 * vertex_adjacency.h
 *
 *  Created on: Jul 23, 2012
 *      Author: hmirap
 */

#ifndef VERTEX_ADJACENCY_H_
#define VERTEX_ADJACENCY_H_

#include <boost/concept_check.hpp>
#include <boost/concept/requires.hpp>

#include "mesh.h"
#include "iterable_vertices.h"

/*! 
 * \struct VertexAdjacencyConcept
 *
 * operations that demanands surrounding 
 * vertices of a vertex
 * 
 * \ingroup concepts
 */
template <class TMesh, class TMesh_Traits = mesh_traits<TMesh>>
struct VertexAdjacencyConcept
{
	typedef typename TMesh_Traits::vertex_descriptor vertex_descriptor;
	typedef typename TMesh_Traits::vv_iterator vv_iterator;

	TMesh m;
	std::pair<vv_iterator,vv_iterator> vvp;
	vertex_descriptor v;
	bool isolated;

	void constraints() {

		boost::function_requires<MeshConcept<TMesh, TMesh_Traits> >();

		vvp = get_adjacent_vertices(m, v);
		isolated = is_isolated(m, v);
	}
};

#endif /* VERTEX_ADJACENCY_H_ */
