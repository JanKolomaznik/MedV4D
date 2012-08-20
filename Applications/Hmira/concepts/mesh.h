/*
 * mesh.h
 *
 *  Created on: Jul 31, 2012
 *      Author: hmirap
 */

#ifndef MESH_H_
#define MESH_H_

#include <boost/config.hpp>
#include <boost/concept_check.hpp>
#include <boost/concept/requires.hpp>


/*!
 * \defgroup concepts
 * Group of concepts that are checked using
 * boost::function_requires
 * 
 * \struct MeshConcept
 * \brief Concept required for a mesh
 * 
 * class is required to contain vertex_descriptor
 * satisfying EqualityComparableConcept and AssignableConcept
 * 
 * \ingroup concepts
 */
template <class TMesh, class TMesh_Traits = mesh_traits<TMesh>>
struct MeshConcept
{
	typedef typename TMesh_Traits::vertex_descriptor vertex_descriptor;

	void constraints() {
	      boost::function_requires< boost::EqualityComparableConcept<vertex_descriptor> >();
	      boost::function_requires< boost::AssignableConcept<vertex_descriptor> >();
	}
};

#endif /* MESH_H_ */
