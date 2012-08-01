/*
 * mesh.h
 *
 *      Author: hmirap
 */

#ifndef MESH_H_
#define MESH_H_

#include <boost/config.hpp>
#include <boost/concept_check.hpp>
#include <boost/concept/requires.hpp>


template <class Mesh>
struct MeshConcept
{
	typedef mesh_traits<Mesh> Mtraits;
	typedef typename Mtraits::vertex_descriptor vertex_descriptor;

	void constraints() {
	      boost::function_requires< boost::EqualityComparableConcept<vertex_descriptor> >();
	      boost::function_requires< boost::AssignableConcept<vertex_descriptor> >();
	}
};

#endif /* MESH_H_ */
