/*
 * winged_edge.h
 *
 *  Created on: Jul 8, 2012
 *      Author: hmirap
 */

#ifndef WINGED_EDGE_H_
#define WINGED_EDGE_H_



#include <vector>
#include <utility>
#include <algorithm>
#include <boost/mpl/bool.hpp>
#include "../traits.h"

/*!
 * \defgroup winged_edge
 * Implementation of Winged edge mesh \link specification http://en.wikipedia.org/wiki/Polygon_mesh#Winged-edge_meshes \endlink
 */
  
/*! \class winged_edge_mesh
 *  \brief Implementation of mesh 
 * 
 * This implementation contains inner classes
 * of vertex, edge, face with proprietary iterators
 * 
 *  \ingroup winged_edge
 */
  template < typename FaceRestriction = triangleMesh>
class winged_edge_mesh {

	public:
		class vertex;
		class edge;
		class face;
		
		class normal;
		
		class vv_iterator;



	public:

		typedef typename FaceRestriction::is_triangle_t is_triangle;

		typedef vertex* vertex_descriptor;
		typedef edge* edge_descriptor;
		typedef face* face_descriptor;

		typedef std::vector<vertex_descriptor> VertexList;
		typedef std::vector<edge_descriptor> EdgeList;
		typedef std::vector<face_descriptor> FaceList;

		typedef typename VertexList::iterator vertex_iterator;
		typedef typename EdgeList::iterator edge_iterator;
		typedef typename FaceList::iterator face_iterator;

		typedef typename VertexList::size_type vertices_size_type;
		typedef typename EdgeList::size_type edges_size_type;
		typedef typename FaceList::size_type faces_size_type;

		typedef typename face::fe_iterator fe_iterator;



	private:
		is_triangle isTriangleMesh;
		VertexList vertices;
		EdgeList edges;
		FaceList faces;

	public:

		winged_edge_mesh();

		~winged_edge_mesh();

		std::pair<vertex_iterator,vertex_iterator> getAllVertices();

		is_triangle isTriangle();

		bool add_vertex(const vertex_descriptor v);

		face_descriptor create_face(
			const vertex_descriptor a,
			const vertex_descriptor b,
			const vertex_descriptor c
			);

		bool create_face(
			const edge_descriptor a,
			const edge_descriptor b,
			const edge_descriptor c);

		void remove_face(const face_descriptor f);

		std::pair<vertex_iterator, vertex_iterator> get_all_vertices();

		std::pair<edge_iterator, edge_iterator> get_all_edges();

		std::pair<face_iterator,face_iterator> get_all_faces();

		std::pair<fe_iterator, fe_iterator> get_surrounding_edges(const face_descriptor f);

	private:
};
  
  /*!
   * \class winged_edge_mesh_traits
   * \brief traits of winged_edge_mesh
   * 
   * \ingroup winged_edge
   * 
   */
template <typename Restriction = triangleMesh>
class winged_edge_mesh_traits
{
public:

	typedef typename winged_edge_mesh<Restriction>::vertex_descriptor vertex_descriptor;
	typedef typename winged_edge_mesh<Restriction>::edge_descriptor edge_descriptor;
	typedef typename winged_edge_mesh<Restriction>::face_descriptor face_descriptor;

	typedef typename winged_edge_mesh<Restriction>::VertexList VertexContainer;
	typedef typename winged_edge_mesh<Restriction>::EdgeList EdgeContainer;
	typedef typename winged_edge_mesh<Restriction>::FaceList FaceContainer;

	typedef typename winged_edge_mesh<Restriction>::vertex_iterator vertex_iterator;
	typedef typename winged_edge_mesh<Restriction>::edge_iterator edge_iterator;
	typedef typename winged_edge_mesh<Restriction>::face_iterator face_iterator;

	typedef typename winged_edge_mesh<Restriction>::vertices_size_type vertices_size_type;
	typedef typename winged_edge_mesh<Restriction>::edges_size_type edges_size_type;
	typedef typename winged_edge_mesh<Restriction>::faces_size_type faces_size_type;

	typedef typename winged_edge_mesh<Restriction>::vv_iterator vv_iterator;
	typedef typename winged_edge_mesh<Restriction>::fe_iterator fe_iterator;



	//==========BASIC CONCEPT=========

	static bool add_vertex(
		typename winged_edge_mesh<Restriction>::vertex_descriptor v,
		class winged_edge_mesh<Restriction> *m);

	static bool create_face(
		typename winged_edge_mesh<Restriction>::vertex_descriptor a,
		typename winged_edge_mesh<Restriction>::vertex_descriptor b,
		typename winged_edge_mesh<Restriction>::vertex_descriptor c,
		class winged_edge_mesh<Restriction> *m);

	static bool remove_face(
		typename winged_edge_mesh<Restriction>::face_descriptor f,
		class winged_edge_mesh<Restriction> *m);

	static std::pair<typename winged_edge_mesh<Restriction>::vertex_iterator,
		typename winged_edge_mesh<Restriction>::vertex_iterator>
	get_all_vertices(const class winged_edge_mesh<Restriction>& m_);

	static std::pair<typename winged_edge_mesh<Restriction>::edge_iterator,
		typename winged_edge_mesh<Restriction>::edge_iterator>
	get_all_edges(const class winged_edge_mesh<Restriction>& m_);

	static bool
	is_isolated(
		class winged_edge_mesh<Restriction>& m_,
		class winged_edge_mesh<Restriction>::vertex* v);


	static std::pair<typename winged_edge_mesh<Restriction>::face_iterator,
		typename winged_edge_mesh<Restriction>::face_iterator>
	get_all_faces(class winged_edge_mesh<Restriction>& m_);

	static std::pair<typename winged_edge_mesh<Restriction>::vv_iterator,
		typename winged_edge_mesh<Restriction>::vv_iterator>
	get_adjacent_vertices(
		class winged_edge_mesh<Restriction>& m_,
		class winged_edge_mesh<Restriction>::vertex* v);




};
  
#include "winged_edge.tcc"
  
/*
  template <typename Restriction>
  std::pair<typename winged_edge_mesh<Restriction>::vv_iterator,
  	  	  	typename winged_edge_mesh<Restriction>::vv_iterator>
  get_adjacent_vertices(
  		  class winged_edge_mesh<Restriction>& m_,
		  class winged_edge_mesh<Restriction>::vertex_descriptor& v_)
  {
	  typedef typename winged_edge_mesh<Restriction>::vertex_descriptor Vertex;
	  Vertex& v = const_cast<Vertex&>(v_);
	  return v->get_adjacent_vertices();
  }*/


#endif /* WINGED_EDGE_H_ */
