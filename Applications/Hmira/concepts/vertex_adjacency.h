/*
 * vertex_adjacency.h
 *
 *      Author: hmirap
 */

#ifndef VERTEX_ADJACENCY_H_
#define VERTEX_ADJACENCY_H_

#include <boost/concept_check.hpp>
#include <boost/concept/requires.hpp>

#include "mesh.h"
#include "iterable_vertices.h"

/*
 * concept pre dotaz na susedov
 *
 * TODO je tu nutné mať iterator cez všetky vrcholy?
 */
template <class Mesh>
struct VertexAdjacencyConcept
{
	typedef mesh_traits<Mesh> Mtraits;
	typedef typename Mtraits::vertex_descriptor vertex_descriptor;
	//typedef typename Mtraits::vertex_iterator vertex_iterator;
	typedef typename Mesh::vv_iterator vv_iterator;

	Mesh m;
	//std::pair<vertex_iterator,vertex_iterator> vp;
	std::pair<vv_iterator,vv_iterator> vvp;
	vertex_descriptor v;
	bool isolated;

	void constraints() {
		//vp = get_all_vertices(m);

		boost::function_requires<MeshConcept<Mesh> >();

		vvp = get_adjacent_vertices(m, v);
		isolated = is_isolated(m, v);
	}
};

#endif /* VERTEX_ADJACENCY_H_ */
