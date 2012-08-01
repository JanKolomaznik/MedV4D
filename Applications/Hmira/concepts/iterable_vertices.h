/*
 * iterable_vertices.h
 *
 *      Author: hmirap
 */

#ifndef ITERABLE_VERTICES_H_
#define ITERABLE_VERTICES_H_

template <class Mesh>
struct IterableVerticesConcept
{
	typedef mesh_traits<Mesh> Mtraits;
	typedef typename Mtraits::vertex_descriptor vertex_descriptor;
	typedef typename Mtraits::vertex_iterator vertex_iterator;

	Mesh m;
	std::pair<vertex_iterator,vertex_iterator> vp;
	vertex_descriptor v;

	void constraints() {
		boost::function_requires<MeshConcept<Mesh> >();

		vp = get_all_vertices(m);
	}
};


#endif /* ITERABLE_VERTICES_H_ */
