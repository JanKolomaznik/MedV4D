/*
 * winged_edge.h
 *
 *      Author: hmirap
 */

#ifndef WINGED_EDGE_H_
#define WINGED_EDGE_H_



#include <vector>
#include <utility>
#include <algorithm>
#include <boost/mpl/bool.hpp>
#include "traits.h"

/*    template <typename M>
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
*/
  /*
   * zatiaľ len pre triangle mesh!!
   * TODO tento prípad podporuje multihrany - ako riešenie halfedges?
   * TODO vyriešiť pre obecný, poradiť sa o koncepte pre faces/edges
   * */


  template < typename FaceRestriction = triangleMesh>
  class winged_edge_mesh {

  public:
	  class vertex;
	  class edge;
	  class face;
	  class vv_iterator;



	  class edge
	  {
	  private:
		  std::pair<vertex*, vertex*> vertices;
		  std::pair<face*, face*> faces;
	  public:
		  edge(vertex* a, vertex* b)
		  {
			  a->add_adjacent_edge(this);
			  b->add_adjacent_edge(this);
			  this->vertices = std::make_pair(a, b);
		  }
		  edge(vertex* a, vertex* b, face* fa, face* fb)
		  {
			  a->add_adjacent_edge(this);
			  b->add_adjacent_edge(this);
			  this->vertices = std::make_pair(a, b);
			  this->faces = std::make_pair(fa, fb);
		  }
		  edge() {};
		  ~edge() {};

		  bool setFace(face* f)
		  {
			  if (!this->faces.first)
				  this->faces.first = f;
			  else
				  this->faces.second = f;

			  return true;
		  }


		  std::pair<vertex*, vertex*> getVertices()
		  {
			  return this->vertices;
		  }
	  };

	  class face
	  {
	  public:
		  typedef typename std::vector<edge*>::iterator fe_iterator;

	  private:
		  std::vector<edge*> cons_edges;

	  public:
		  face() {};
		  face(edge* a, edge* b, edge* c)
		  {
			  this->cons_edges.push_back(a);
			  this->cons_edges.push_back(b);
			  this->cons_edges.push_back(c);
		  };
		  ~face() {};

		  std::pair<fe_iterator, fe_iterator>
		  	  get_edges() { return std::make_pair(this->cons_edges.begin(), this->cons_edges.end());}

	  };

	  template <typename FaceRes>
	  class winged_edge_mesh_traits
	  {

		  typedef typename FaceRes::is_triangle_t is_triangle;
	  public:

		typedef vertex* vertex_descriptor;
		typedef edge* edge_descriptor;
	    typedef face* face_descriptor;

	    typedef typename face::fe_iterator fe_iterator;
	  };


  public:

	  typedef typename FaceRestriction::is_triangle_t is_triangle;

	  typedef typename winged_edge_mesh_traits<FaceRestriction>::vertex_descriptor vertex_descriptor;
	  typedef typename winged_edge_mesh_traits<FaceRestriction>::edge_descriptor edge_descriptor;
	  typedef typename winged_edge_mesh_traits<FaceRestriction>::face_descriptor face_descriptor;

	  typedef std::vector<vertex_descriptor> VertexList;
	  typedef std::vector<edge_descriptor> EdgeList;
	  typedef std::vector<face_descriptor> FaceList;

	  typedef typename VertexList::iterator vertex_iterator;
	  typedef typename EdgeList::iterator edge_iterator;
	  typedef typename FaceList::iterator face_iterator;

	  typedef typename VertexList::size_type vertices_size_type;
	  typedef typename EdgeList::size_type edges_size_type;
	  typedef typename FaceList::size_type faces_size_type;

	  typedef typename winged_edge_mesh_traits<FaceRestriction>::fe_iterator fe_iterator;

	  class vertex
	  {
	  private:
		  std::size_t id;
		  std::vector<edge*> adjacent_edges;
	  public:
		  vertex(){ this->id = -1; }
		  vertex(std::size_t id) { this->id = id; }
		  ~vertex(){}
		  bool add_adjacent_edge(edge* e)
		  {
			  this->adjacent_edges.push_back(e);
			  return true;
		  }
		  std::size_t get_id() { return this->id; }

		  bool is_isolated()
		  {
			  return (this->adjacent_edges.empty());
		  }

		  std::pair<vv_iterator, vv_iterator> get_adjacent_vertices()
		{
			  vertex* fst_vertex;
			  if ((*adjacent_edges.begin())->getVertices().first == this)
				  fst_vertex = (*adjacent_edges.begin())->getVertices().second;
			  else
				  fst_vertex = (*adjacent_edges.begin())->getVertices().first;
			  vv_iterator a1(adjacent_edges.begin(), adjacent_edges.end(), this, fst_vertex);
			  vv_iterator a2(adjacent_edges.end(), adjacent_edges.end(), this, NULL);
			  return std::make_pair(a1, a2);
		}

		  std::pair<edge_iterator, edge_iterator> get_adjacent_edges()
		{
			  return std::make_pair(this->adjacent_edges.begin(), this->adjacent_edges.end());
		}
	  };

	  class vv_iterator : public std::iterator<std::input_iterator_tag, vertex_descriptor>
	  {
	  public:
		  vertex* sender;
		  vertex* p;
		  edge_iterator e_i;
		  edge_iterator edges_end;

	  public:
		vv_iterator(const vv_iterator& mit) : p(mit.p)
		{
			this->sender = mit.sender;
			this->p = mit.p;
			this->e_i = mit.e_i;
			this->edges_end = mit.edges_end;
		}
		vv_iterator(edge_iterator ei, edge_iterator edges_end, vertex* sender, vertex* p)
		{
			this->edges_end = edges_end;
			this->e_i = ei;
			this->sender = sender;
			this->p = p;
		}
	    vv_iterator& operator++()
	    {
	    	++e_i;
	    	//if (*e_i == NULL)
			if (e_i == edges_end)
							return *this;
	    	vertex* v1 = ((*e_i)->getVertices().first == this->sender) ?
	    			(*e_i)->getVertices().second : (*e_i)->getVertices().first;
	    	p = v1;
	    	return *this;
	    }
	    bool operator==(const vv_iterator& rhs) {return e_i==rhs.e_i;}
	    bool operator!=(const vv_iterator& rhs) {return e_i!=rhs.e_i;}
	    vertex_descriptor& operator*() {return p;}
	  };

  private:
	  is_triangle isTriangleMesh;
	  VertexList vertices;
	  EdgeList edges;
	  FaceList faces;

  public:

	  winged_edge_mesh()
	  {
	  }

	  ~winged_edge_mesh()
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

	  face_descriptor create_face(
			  const vertex_descriptor a,
			  const vertex_descriptor b,
			  const vertex_descriptor c
			  )
	  {
		  if (std::find(vertices.begin(), vertices.end(), a) == vertices.end())
			  return false;
		  if (std::find(vertices.begin(), vertices.end(), b) == vertices.end())
			  return false;
		  if (std::find(vertices.begin(), vertices.end(), c) == vertices.end())
			  return false;
		  edge* ea = new edge(a,b);
		  edge* eb = new edge(b,c);
		  edge* ec = new edge(c,a);


		  face* f = new face(ea, eb, ec);

		  ea->setFace(f);
		  eb->setFace(f);
		  ec->setFace(f);

		  this->edges.push_back(ea);
		  this->edges.push_back(eb);
		  this->edges.push_back(ec);

		  this->faces.push_back(f);

		  return f;
	  }

	  bool create_face(
			  const edge_descriptor a,
			  const edge_descriptor b,
			  const edge_descriptor c)
	  {
		  if (std::find(edges.begin(), edges.end(), a) == edges.end())
			  return false;
		  if (std::find(edges.begin(), edges.end(), b) == edges.end())
			  return false;
		  if (std::find(edges.begin(), edges.end(), c) == edges.end())
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

	  std::pair<edge_iterator, edge_iterator> get_all_edges()
	{
		  return std::make_pair(edges.begin(), edges.end());
	}

	  std::pair<face_iterator,face_iterator> get_all_faces()
	  {
		  return std::make_pair(faces.begin(), faces.end());
	  }

	  std::pair<fe_iterator, fe_iterator> get_surrounding_edges(const face_descriptor f)
	  {
		  return f->get_edges();
	  }

  private:
  };

  //==========BASIC CONCEPT=========

  template <typename Restriction>
  bool add_vertex(
		  	  	  typename winged_edge_mesh<Restriction>::vertex_descriptor v,
		  	  	  class winged_edge_mesh<Restriction> *m)
  {
	  return  m->add_vertex(v);
  }

  template <typename Restriction>
  bool create_face(
				  typename winged_edge_mesh<Restriction>::vertex_descriptor a,
				  typename winged_edge_mesh<Restriction>::vertex_descriptor b,
				  typename winged_edge_mesh<Restriction>::vertex_descriptor c,
		  	  	  class winged_edge_mesh<Restriction> *m)
  {
	  return  m->create_face(a,b,c);
  }

  template <typename Restriction>
  bool remove_face(
				  typename winged_edge_mesh<Restriction>::face_descriptor f,
		  	  	  class winged_edge_mesh<Restriction> *m)
  {
	  return  m->remove_face(f);
  }

  template <typename Restriction>
  std::pair<typename winged_edge_mesh<Restriction>::vertex_iterator,
  	  	  	typename winged_edge_mesh<Restriction>::vertex_iterator>
  get_all_vertices(const class winged_edge_mesh<Restriction>& m_)
  {
	  typedef winged_edge_mesh<Restriction> Mesh;
	  Mesh& m = const_cast<Mesh&>(m_);
	  return m.get_all_vertices();
  }

  template <typename Restriction>
  std::pair<typename winged_edge_mesh<Restriction>::edge_iterator,
  	  	  	typename winged_edge_mesh<Restriction>::edge_iterator>
  get_all_edges(const class winged_edge_mesh<Restriction>& m_)
  {
	  typedef winged_edge_mesh<Restriction> Mesh;
	  Mesh& m = const_cast<Mesh&>(m_);
	  return m.get_all_edges();
  }

  template <typename Restriction>
  bool
  is_isolated(
		  class winged_edge_mesh<Restriction>& m_,
		  class winged_edge_mesh<Restriction>::vertex* v)
  {
	  return v->is_isolated();
  }


  template <typename Restriction>
  std::pair<typename winged_edge_mesh<Restriction>::face_iterator,
  	  	  	typename winged_edge_mesh<Restriction>::face_iterator>
  get_all_faces(class winged_edge_mesh<Restriction>& m_)
  {
	  typedef winged_edge_mesh<Restriction> Mesh;
	  Mesh& m = const_cast<Mesh&>(m_);
	  return m.get_all_faces();
  }

  template <typename Restriction>
  std::pair<typename winged_edge_mesh<Restriction>::vv_iterator,
  	  	  	typename winged_edge_mesh<Restriction>::vv_iterator>
  get_adjacent_vertices(
		  class winged_edge_mesh<Restriction>& m_,
		  class winged_edge_mesh<Restriction>::vertex* v)
  {
	  return v->get_adjacent_vertices();
  }

/*
 * TODO nejde a neviem prečo
 *
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
