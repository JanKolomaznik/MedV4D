/*
 * OpenMeshX.h
 *
 *  Created on: Jul 16, 2012
 *      Author: hmirap
 */

#ifndef OPENMESHX_H_
#define OPENMESHX_H_

#include <OpenMesh/Core/Mesh/PolyMesh_ArrayKernelT.hh>
#include <OpenMesh/Core/IO/MeshIO.hh>
#include <vector>


typedef OpenMesh::PolyMesh_ArrayKernelT<> OpenMeshExtended;

/*!
 * \defgroup OpenMeshX
 * implementation of polynomial mesh with all features of OpenMesh
 * more: \link OpenMesh http://openmesh.org/ \endlink 
 * 
 * \class OpenMeshX
 * \brief class inherited from OpenMesh::PolyMesh_ArrayKernelT
 * 
 * \ingroup OpenMeshX
 */
class OpenMeshX : public OpenMeshExtended
{
public:

	typedef typename OpenMeshExtended::Point Point;


    typedef typename OpenMeshExtended::VertexHandle vertex_descriptor;
    typedef typename OpenMeshExtended::VertexIter vertex_iterator;
    typedef typename std::vector<OpenMeshExtended::Vertex>::size_type vertices_size_type;

    typedef typename OpenMeshExtended::EdgeHandle edge_descriptor;
    typedef typename OpenMeshExtended::EdgeIter edge_iterator;
    typedef typename std::vector<OpenMeshExtended::Edge>::size_type edges_size_type;

    typedef typename OpenMeshExtended::FaceHandle face_descriptor;
    typedef typename OpenMeshExtended::FaceIter face_iterator;
    typedef typename std::vector<OpenMeshExtended::Face>::size_type faces_size_type;

    typedef typename OpenMeshExtended::FVIter fv_iterator;
    typedef typename OpenMeshExtended::VVIter vv_iterator;
    typedef typename OpenMeshExtended::VEIter ve_iterator;

    std::pair<fv_iterator, fv_iterator>
    get_surrounding_vertices(face_descriptor fd)
    {
    	return std::make_pair(this->fv_begin(fd), this->fv_end(fd));
    }


};

/*!
 * \struct OpenMeshXTraits
 * \brief traits of OpenMeshX
 * 
 * \ingroup OpenMeshX
 */
struct OpenMeshXTraits : OpenMesh::DefaultTraits
{
  typedef typename OpenMeshX::vertex_descriptor vertex_descriptor;
  typedef typename OpenMeshX::edge_descriptor edge_descriptor;
  typedef typename OpenMeshX::face_descriptor face_descriptor;
  
  typedef typename OpenMeshX::vertex_iterator vertex_iterator;
  typedef typename OpenMeshX::edge_iterator edge_iterator;
  typedef typename OpenMeshX::face_iterator face_iterator;
  
  typedef typename OpenMeshX::vv_iterator vv_iterator;
  typedef typename OpenMeshX::ve_iterator ve_iterator;
  typedef typename OpenMeshX::fv_iterator fv_iterator;
  
  typedef typename OpenMeshX::vertices_size_type vertices_size_type;
  typedef typename OpenMeshX::edges_size_type edges_size_type;
  typedef typename OpenMeshX::faces_size_type faces_size_type;
  
};

inline OpenMeshX::vertex_descriptor operator*(OpenMeshX::fv_iterator fvi) { return fvi.handle(); }
inline OpenMeshX::vertex_descriptor operator*(OpenMeshX::vv_iterator vvi) { return vvi.handle(); }
//inline OpenMeshX::vertex_descriptor* operator->(OpenMeshX::fv_iterator fvi) { return &fvi.handle(); }
//TODO toto zase prečo nejde!?

//=================CONCEPTS======================

bool remove_vertex(
		  	  	  typename OpenMeshXTraits::vertex_descriptor v,
		  	  	  class OpenMeshX *m)
{
	 m->delete_vertex(v);
	 return true;
}

bool create_face(
				  typename OpenMeshXTraits::vertex_descriptor a,
				  typename OpenMeshXTraits::vertex_descriptor b,
				  typename OpenMeshXTraits::vertex_descriptor c,
		  	  	  class OpenMeshX *m)
{
	std::vector<OpenMeshXTraits::vertex_descriptor>  face_vhandles;

	face_vhandles.clear();
	face_vhandles.push_back(a);
	face_vhandles.push_back(b);
	face_vhandles.push_back(c);
	m->add_face(face_vhandles);
	return true;
}

bool remove_face(
				  typename OpenMeshXTraits::face_descriptor f,
		  	  	  class OpenMeshX *m)
{
	  m->delete_face(f);
	  return true;
}

std::pair<typename OpenMeshXTraits::vertex_iterator,
	  	  	typename OpenMeshXTraits::vertex_iterator>
get_all_vertices(const class OpenMeshX& m_)
{
	  typedef OpenMeshX Mesh;
	  Mesh& m = const_cast<Mesh&>(m_);
	  return std::make_pair(m.vertices_begin(), m.vertices_end());
}

std::pair<typename OpenMeshXTraits::edge_iterator,
	  	  	typename OpenMeshXTraits::edge_iterator>
get_all_edges(const class OpenMeshX& m_)
{
	  typedef OpenMeshX Mesh;
	  Mesh& m = const_cast<Mesh&>(m_);
	  return std::make_pair(m.edges_begin(), m.edges_end());
}


std::pair<typename OpenMeshXTraits::face_iterator,
	  	  	typename OpenMeshXTraits::face_iterator>
get_all_faces(const class OpenMeshX& m_)
{
	  typedef OpenMeshX Mesh;
	  Mesh& m = const_cast<Mesh&>(m_);
	  return std::make_pair(m.faces_begin(), m.faces_end());
}

//=========== VERTEX ADJACENCY CONCEPT ===========

bool is_isolated(const class OpenMeshX& m_,
		OpenMeshX::vertex_descriptor v)
{
	  typedef OpenMeshX Mesh;
	  Mesh& m = const_cast<Mesh&>(m_);
	  return m.is_isolated(v);
}

std::pair<typename OpenMeshXTraits::vv_iterator,
	  	  	typename OpenMeshXTraits::vv_iterator>
get_adjacent_vertices(
		const class OpenMeshX& m_,
		  OpenMeshXTraits::vertex_descriptor v)
		  {
	  typedef OpenMeshX Mesh;
	  Mesh& m = const_cast<Mesh&>(m_);
	  return std::make_pair(m.vv_begin(v), m.vv_end(v));
		  }

std::pair<typename OpenMeshXTraits::ve_iterator,typename OpenMeshXTraits::ve_iterator>
get_adjacent_edges(
		const class OpenMeshX& m_,
		OpenMeshXTraits::vertex_descriptor v)
		  {

	  typedef OpenMeshX Mesh;
	  Mesh& m = const_cast<Mesh&>(m_);
	  return std::make_pair(m.ve_begin(v),m.ve_end(v));
		  }

		  
#endif /* OPENMESHX_H_ */
