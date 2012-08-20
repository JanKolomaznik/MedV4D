/*
 * traits.h
 *
 *  Created on: Jun 6, 2012
 *      Author: hmirap
 */

#ifndef TRAITS_H_
#define TRAITS_H_


#include <vector>
#include <utility>
#include <algorithm>
#include <boost/mpl/bool.hpp>

    template <typename M>
    struct mesh_traits {
        typedef typename M::vertex_descriptor      vertex_descriptor;
        typedef typename M::vertices_size_type     vertices_size_type;
        typedef typename M::vertex_iterator        vertex_iterator;

        static inline vertex_descriptor null_vertex();
    };

  struct triangleMesh
  {
	  enum { is_triangle = true};
	  typedef boost::mpl::true_ is_triangle_t;
  };

  struct polynomialMesh
  {
	  enum { is_triangle = false};
	  typedef boost::mpl::false_ is_triangle_t;
  };

  /*
   * zatiaľ len pre triangle mesh!!
   * TODO vyriešiť pre obecný, poradiť sa o koncepte pre faces/edges
   * */

  template < typename FaceRestriction = triangleMesh>
  class vf_mesh {


	  /*
	   * zatiaľ len pre triangle mesh!!
	   * TODO vyriešiť pre obecný, poradiť sa o koncepte pre faces/edges
	   * */
	  class face
	  {
	  public:
		  face() {};
		  face(std::size_t a, std::size_t b, std::size_t c)
		  {
			  this->cons_vertices.push_back(a);
			  this->cons_vertices.push_back(b);
			  this->cons_vertices.push_back(c);
		  };
		  ~face() {};
		  std::pair<std::vector<size_t>::iterator, std::vector<size_t>::iterator>
		  	  get_vertices() { return std::make_pair(this->cons_vertices.begin(), this->cons_vertices.end());}

	  public:
		  std::vector<std::size_t> cons_vertices;
	  };

	  template <typename FaceRes>
	  class vf_mesh_traits
	  {

		  typedef typename FaceRes::is_triangle_t is_triangle;
	  public:

	    typedef std::size_t vertex_descriptor;
	    typedef face face_descriptor;
	  };


  public:

	  typedef typename FaceRestriction::is_triangle_t is_triangle;

	  typedef typename vf_mesh_traits<FaceRestriction>::vertex_descriptor vertex_descriptor;
	  typedef typename vf_mesh_traits<FaceRestriction>::face_descriptor face_descriptor;

	  typedef std::vector<vertex_descriptor> VertexList;
	  typedef std::vector<face_descriptor> FaceList;

	  typedef typename VertexList::iterator vertex_iterator;
	  typedef typename FaceList::iterator face_iterator;

	  typedef typename VertexList::size_type vertices_size_type;
	  typedef typename FaceList::size_type faces_size_type;



  private:
	  is_triangle isTriangleMesh;
	  VertexList vertices;
	  FaceList faces;

  public:

	  vf_mesh()
	  {
	  }

	  ~vf_mesh()
	  {
	  }

	  std::pair<vertex_iterator,vertex_iterator> getAllVertices()
	  {
		  return std::make_pair(vertices.begin(), vertices.end());
	  }
	  is_triangle isTriangle()
	  {
		  return isTriangleMesh;
	  }

	  bool add_vertex(const vertex_descriptor v)
	  {
		  if (std::find(vertices.begin(), vertices.end(), v) != vertices.end())
			  return false;
		  this->vertices.push_back(v);
		  return true;
	  }

	  bool create_face(
			  const vertex_descriptor a,
			  const vertex_descriptor b,
			  const vertex_descriptor c)
	  {
		  if (std::find(vertices.begin(), vertices.end(), a) == vertices.end())
			  return false;
		  if (std::find(vertices.begin(), vertices.end(), b) == vertices.end())
			  return false;
		  if (std::find(vertices.begin(), vertices.end(), c) == vertices.end())
			  return false;
		  face F(a,b,c);
		  this->faces.push_back(F);
		  return true;
	  }

	  void remove_face(const face_descriptor f)
	  {
		  face_iterator f_iter = std::find(faces.begin(), faces.end(), f);
		  if (f == faces.end())
			  return;
		  this->faces.erase(f_iter);
	  }

	  std::pair<vertex_iterator, vertex_iterator> get_all_vertices()
	  {
		  return std::make_pair(vertices.begin(), vertices.end());
	  }

	  std::pair<face_iterator,face_iterator> get_all_faces()
	  {
		  return std::make_pair(faces.begin(), faces.end());
	  }

  private:
  };

  //==========BASIC CONCEPT=========

  template <typename Restriction>
  bool add_vertex(
		  	  	  typename vf_mesh<Restriction>::vertex_descriptor v,
		  	  	  class vf_mesh<Restriction> *m)
  {
	  return  m->add_vertex(v);
  }

  template <typename Restriction>
  bool create_face(
				  typename vf_mesh<Restriction>::vertex_descriptor a,
				  typename vf_mesh<Restriction>::vertex_descriptor b,
				  typename vf_mesh<Restriction>::vertex_descriptor c,
		  	  	  class vf_mesh<Restriction> *m)
  {
	  return  m->create_face(a,b,c);
  }

  template <typename Restriction>
  bool remove_face(
				  typename vf_mesh<Restriction>::face_descriptor f,
		  	  	  class vf_mesh<Restriction> *m)
  {
	  return  m->remove_face(f);
  }

  template <typename Restriction>
  std::pair<typename vf_mesh<Restriction>::vertex_iterator,
  	  	  	typename vf_mesh<Restriction>::vertex_iterator>
  get_all_vertices(const class vf_mesh<Restriction>& m_)
  {
	  typedef vf_mesh<Restriction> Mesh;
	  Mesh& m = const_cast<Mesh&>(m_);
	  return m.get_all_vertices();
  }

  template <typename Restriction>
  std::pair<typename vf_mesh<Restriction>::face_iterator,
  	  	  	typename vf_mesh<Restriction>::face_iterator>
  get_all_faces(class vf_mesh<Restriction>& m_)
  {
	  typedef vf_mesh<Restriction> Mesh;
	  Mesh& m = const_cast<Mesh&>(m_);
	  return m.get_all_faces();
  }


#endif /* TRAITS_H_ */
