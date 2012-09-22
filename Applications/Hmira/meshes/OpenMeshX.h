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

struct MyTraits : public OpenMesh::DefaultTraits
{
  VertexAttributes(OpenMesh::Attributes::Status);
  FaceAttributes(OpenMesh::Attributes::Status);
  EdgeAttributes(OpenMesh::Attributes::Status);
};


typedef OpenMesh::PolyMesh_ArrayKernelT<MyTraits> OpenMeshExtended;

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
/*class OpenMeshX : public OpenMeshExtended
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
    get_surrounding_vertices(face_descriptor fd);


};*/

/*!
 * \struct OpenMeshXTraits
 * \brief traits of OpenMeshX
 * 
 * \ingroup OpenMeshX
 */
template<>
class mesh_traits<OpenMeshExtended>

{
public:
	typedef OpenMesh::DefaultTraits original_traits;

	typedef typename OpenMeshExtended::Point Point;

	typedef typename OpenMeshExtended::Normal normal;

	typedef typename OpenMeshExtended::VertexHandle vertex_descriptor;
	typedef typename OpenMeshExtended::VertexIter vertex_iterator;
	typedef typename std::vector<OpenMeshExtended::Vertex>::size_type vertices_size_type;

	typedef typename OpenMeshExtended::EdgeHandle edge_descriptor;
	typedef typename OpenMeshExtended::EdgeIter edge_iterator;
	typedef typename std::vector<OpenMeshExtended::Edge>::size_type edges_size_type;

	typedef typename OpenMeshExtended::FaceHandle face_descriptor;
	typedef typename OpenMeshExtended::FaceIter face_iterator;
	typedef typename std::vector<OpenMeshExtended::Face>::size_type faces_size_type;

	typedef typename OpenMeshExtended::VEIter ve_iterator;	


	
	class my_fv_iterator : public OpenMesh::Iterators::FaceVertexIterT< OpenMesh::PolyConnectivity >
	{
	public:	
		typedef OpenMesh::PolyConnectivity& mesh_ref;

/* 	TODO prekonzultovať 
*	constructor inheritance v norme c++0x 
*/

		my_fv_iterator() : 
			OpenMesh::Iterators::FaceVertexIterT< OpenMesh::PolyConnectivity >() {};
		
		my_fv_iterator (mesh_ref _mesh, OpenMesh::PolyConnectivity::FaceHandle _start, bool _end) :
			OpenMesh::Iterators::FaceVertexIterT< OpenMesh::PolyConnectivity >(_mesh, _start, _end) {};

		my_fv_iterator(mesh_ref _mesh, OpenMesh::PolyConnectivity::HalfedgeHandle _heh, bool _end) : 
			OpenMesh::Iterators::FaceVertexIterT< OpenMesh::PolyConnectivity >(_mesh, _heh, _end) {};

		my_fv_iterator(const OpenMesh::Iterators::FaceVertexIterT< OpenMesh::PolyConnectivity >& _rhs) :
			OpenMesh::Iterators::FaceVertexIterT< OpenMesh::PolyConnectivity >(_rhs) {};

		bool operator==(const my_fv_iterator& _rhs) const 
		{
			return 
			((mesh_   == _rhs.mesh_) &&
			(start_  == _rhs.start_) &&
			(heh_    == _rhs.heh_));
		}
 
 
		bool operator!=(const my_fv_iterator& _rhs) const
		{
			return !operator==(_rhs);
		}


	};

	class my_vv_iterator : public OpenMesh::Iterators::VertexVertexIterT< OpenMesh::PolyConnectivity >
	{
	public:	
		typedef OpenMesh::PolyConnectivity& mesh_ref;

		my_vv_iterator() : 
			OpenMesh::Iterators::VertexVertexIterT< OpenMesh::PolyConnectivity >() {};
		
		my_vv_iterator (mesh_ref _mesh, OpenMesh::PolyConnectivity::VertexHandle _start, bool _end) :
			OpenMesh::Iterators::VertexVertexIterT< OpenMesh::PolyConnectivity >(_mesh, _start, _end) {};

		my_vv_iterator(mesh_ref _mesh, OpenMesh::PolyConnectivity::HalfedgeHandle _heh, bool _end) : 
			OpenMesh::Iterators::VertexVertexIterT< OpenMesh::PolyConnectivity >(_mesh, _heh, _end) {};

		my_vv_iterator(const OpenMesh::Iterators::VertexVertexIterT< OpenMesh::PolyConnectivity >& _rhs) :
			OpenMesh::Iterators::VertexVertexIterT< OpenMesh::PolyConnectivity >(_rhs) {};

		bool operator==(const my_vv_iterator& _rhs) const 
		{
			return 
			((mesh_   == _rhs.mesh_) &&
			(start_  == _rhs.start_) &&
			(heh_    == _rhs.heh_));
		}
 
 
		bool operator!=(const my_vv_iterator& _rhs) const
		{
			return !operator==(_rhs);
		}


	};




	class my_ve_iterator : public OpenMeshExtended::VEIter
	{
		
	};

	typedef my_fv_iterator fv_iterator;
	typedef my_vv_iterator vv_iterator;

    static std::pair<fv_iterator, fv_iterator>
    get_surrounding_vertices(const OpenMeshExtended& m_, face_descriptor fd);


//=================CONCEPTS======================

static bool remove_vertex(
					  vertex_descriptor v,
		  	  	  OpenMeshExtended &m);

static bool create_face(
				  vertex_descriptor a,
				  vertex_descriptor b,
				  vertex_descriptor c,
		  	  	  OpenMeshExtended *m);

static bool remove_face(
				  typename mesh_traits<OpenMeshExtended>::face_descriptor f,
		  	  	  OpenMeshExtended* m);

static std::pair<vertex_iterator,
	  	  	vertex_iterator>
get_all_vertices(const OpenMeshExtended& m_);

static std::pair<edge_iterator,
	  	  	edge_iterator>
get_all_edges(const OpenMeshExtended& m_);


static std::pair<face_iterator,
	  	  	face_iterator>
get_all_faces(const OpenMeshExtended& m_);

//=========== VERTEX ADJACENCY CONCEPT ===========

static bool is_isolated(const OpenMeshExtended& m_,
		vertex_descriptor v);

static std::pair<vv_iterator,
	  	  vv_iterator>
get_adjacent_vertices(
		const OpenMeshExtended& m_,
		  vertex_descriptor v);

static std::pair<ve_iterator, ve_iterator>
get_adjacent_edges(
		const OpenMeshExtended& m_,
		vertex_descriptor v);


 
};

template<>
class advanced_mesh_traits<OpenMeshExtended> : public mesh_traits<OpenMeshExtended>
{
public:
	static mesh_traits<OpenMeshExtended>::normal 
	get_face_normal(
		const OpenMeshExtended& m_,
		face_descriptor f);

	static bool
	flip_face_normal(
		OpenMeshExtended& m_,
		face_descriptor& f);
};



typedef mesh_traits<OpenMeshExtended> OpenMeshXTraits;

inline mesh_traits<OpenMeshExtended>::vertex_descriptor operator*(mesh_traits<OpenMeshExtended>::fv_iterator fvi) { return fvi.handle(); }
inline mesh_traits<OpenMeshExtended>::vertex_descriptor operator*(mesh_traits<OpenMeshExtended>::vv_iterator vvi) { return vvi.handle(); }
#include "OpenMeshX.tcc"
		  
#endif /* OPENMESHX_H_ */
