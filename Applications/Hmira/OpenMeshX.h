/*
 * OpenMeshX.h
 *
 *      Author: hmirap
 */

#ifndef OPENMESHX_H_
#define OPENMESHX_H_

#include <OpenMesh/Core/Mesh/PolyMesh_ArrayKernelT.hh>
#include <OpenMesh/Core/IO/MeshIO.hh>
#include <vector>


typedef OpenMesh::PolyMesh_ArrayKernelT<> OpenMeshExtended;

struct OpenMeshXTraits : OpenMesh::DefaultTraits
{
};

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

inline OpenMeshX::vertex_descriptor operator*(OpenMeshX::fv_iterator fvi) { return fvi.handle(); }
inline OpenMeshX::vertex_descriptor operator*(OpenMeshX::vv_iterator vvi) { return vvi.handle(); }
//inline OpenMeshX::vertex_descriptor* operator->(OpenMeshX::fv_iterator fvi) { return &fvi.handle(); }
//TODO toto zase prečo nejde!?

//=================CONCEPTS======================

bool remove_vertex(
		  	  	  typename OpenMeshX::vertex_descriptor v,
		  	  	  class OpenMeshX *m)
{
	 m->delete_vertex(v);
	 return true;
}

bool create_face(
				  typename OpenMeshX::vertex_descriptor a,
				  typename OpenMeshX::vertex_descriptor b,
				  typename OpenMeshX::vertex_descriptor c,
		  	  	  class OpenMeshX *m)
{
	std::vector<OpenMeshX::vertex_descriptor>  face_vhandles;

	face_vhandles.clear();
	face_vhandles.push_back(a);
	face_vhandles.push_back(b);
	face_vhandles.push_back(c);
	m->add_face(face_vhandles);
	return true;
}

bool remove_face(
				  typename OpenMeshX::face_descriptor f,
		  	  	  class OpenMeshX *m)
{
	  m->delete_face(f);
	  return true;
}

std::pair<typename OpenMeshX::vertex_iterator,
	  	  	typename OpenMeshX::vertex_iterator>
get_all_vertices(const class OpenMeshX& m_)
{
	  typedef OpenMeshX Mesh;
	  Mesh& m = const_cast<Mesh&>(m_);
	  return std::make_pair(m.vertices_begin(), m.vertices_end());
}

std::pair<typename OpenMeshX::edge_iterator,
	  	  	typename OpenMeshX::edge_iterator>
get_all_edges(const class OpenMeshX& m_)
{
	  typedef OpenMeshX Mesh;
	  Mesh& m = const_cast<Mesh&>(m_);
	  return std::make_pair(m.edges_begin(), m.edges_end());
}


std::pair<typename OpenMeshX::face_iterator,
	  	  	typename OpenMeshX::face_iterator>
get_all_faces(const class OpenMeshX& m_)
{
	  typedef OpenMeshX Mesh;
	  Mesh& m = const_cast<Mesh&>(m_);
	  return std::make_pair(m.faces_begin(), m.faces_end());
}

//=========== VERTEX ADJACENCY CONCEPT ===========
/*
bool is_isolated(const class OpenMeshX& m_,
		OpenMeshX::vertex_descriptor v)
{
	  typedef OpenMeshX Mesh;
	  Mesh& m = const_cast<Mesh&>(m_);
	  return m.is_isolated(v);
}*/

std::pair<typename OpenMeshX::vv_iterator,
	  	  	typename OpenMeshX::vv_iterator>
get_adjacent_vertices(
		const class OpenMeshX& m_,
		  OpenMeshX::vertex_descriptor v)
		  {
	  typedef OpenMeshX Mesh;
	  Mesh& m = const_cast<Mesh&>(m_);
	  return std::make_pair(m.vv_begin(v), m.vv_end(v));
		  }

std::pair<typename OpenMeshX::ve_iterator,typename OpenMeshX::ve_iterator>
get_adjacent_edges(
		const class OpenMeshX& m_,
		OpenMeshX::vertex_descriptor v)
		  {

	  typedef OpenMeshX Mesh;
	  Mesh& m = const_cast<Mesh&>(m_);
	  return std::make_pair(m.ve_begin(v),m.ve_end(v));
		  }

#endif /* OPENMESHX_H_ */
