	  
	  template < typename FaceRestriction>
	  class winged_edge_mesh<FaceRestriction>::vertex
	  {
	  private:
		  std::size_t id;
		  std::vector<winged_edge_mesh<FaceRestriction>::edge*> adjacent_edges;
	  public:
		  vertex(){ this->id = -1; }
		  vertex(std::size_t id) { this->id = id; }
		  ~vertex(){}
		  bool add_adjacent_edge(winged_edge_mesh<FaceRestriction>::edge* e)
		  {
			  this->adjacent_edges.push_back(e);
			  return true;
		  }
		  std::size_t get_id() { return this->id; }

		  bool is_isolated()
		  {
			  return (this->adjacent_edges.empty());
		  }

		std::pair<winged_edge_mesh<FaceRestriction>::vv_iterator, winged_edge_mesh<FaceRestriction>::vv_iterator> get_adjacent_vertices()
		{

			if (is_isolated())
			{
				winged_edge_mesh<FaceRestriction>::vv_iterator a(adjacent_edges.begin());
				return std::make_pair(a,a);
			}
 
			vertex* fst_vertex;

			if ((*adjacent_edges.begin())->getVertices().first == this)
				  fst_vertex = (*adjacent_edges.begin())->getVertices().second;
			else
				  fst_vertex = (*adjacent_edges.begin())->getVertices().first;

			winged_edge_mesh<FaceRestriction>::vv_iterator a1(adjacent_edges.begin(), adjacent_edges.end(), this, fst_vertex);
			winged_edge_mesh<FaceRestriction>::vv_iterator a2(adjacent_edges.end(), adjacent_edges.end(), this, NULL);
			
			return std::make_pair(a1, a2);
		}

		  std::pair<winged_edge_mesh<FaceRestriction>::edge_iterator, winged_edge_mesh<FaceRestriction>::edge_iterator> get_adjacent_edges()
		{
			  return std::make_pair(this->adjacent_edges.begin(), this->adjacent_edges.end());
		}
	  };


    template < typename FaceRestriction>
    class winged_edge_mesh<FaceRestriction>::edge
    {
    private:
	    std::pair<winged_edge_mesh<FaceRestriction>::vertex*, winged_edge_mesh<FaceRestriction>::vertex*> vertices;
	    std::pair<winged_edge_mesh<FaceRestriction>::face*, winged_edge_mesh<FaceRestriction>::face*> faces;
    public:
	    edge(winged_edge_mesh<FaceRestriction>::vertex* a, winged_edge_mesh<FaceRestriction>::vertex* b)
	    {
		    a->add_adjacent_edge(this);
		    b->add_adjacent_edge(this);
		    this->vertices = std::make_pair(a, b);
	    }
	    edge(
	    winged_edge_mesh<FaceRestriction>::vertex* a, 
	    winged_edge_mesh<FaceRestriction>::vertex* b, 
	    winged_edge_mesh<FaceRestriction>::face* fa, 
	    winged_edge_mesh<FaceRestriction>::face* fb)
	    {
		    a->add_adjacent_edge(this);
		    b->add_adjacent_edge(this);
		    this->vertices = std::make_pair(a, b);
		    this->faces = std::make_pair(fa, fb);
	    }
	    edge() {};
	    ~edge() {};

	    bool setFace(winged_edge_mesh<FaceRestriction>::face* f)
	    {
		    if (!this->faces.first)
			    this->faces.first = f;
		    else
			    this->faces.second = f;

		    return true;
	    }


	    std::pair<winged_edge_mesh<FaceRestriction>::vertex*, winged_edge_mesh<FaceRestriction>::vertex*> getVertices()
	    {
		    return this->vertices;
	    }
    };
    
    template < typename FaceRestriction>
    class winged_edge_mesh<FaceRestriction>::face
    {
	public:
		typedef typename std::vector<winged_edge_mesh<FaceRestriction>::edge*>::iterator fe_iterator;
		winged_edge_mesh<FaceRestriction>::normal face_normal;


    private:
	    std::vector<winged_edge_mesh<FaceRestriction>::edge*> cons_edges;

    public:
	    face() {};
	    face(
	      winged_edge_mesh<FaceRestriction>::edge* a, 
	      winged_edge_mesh<FaceRestriction>::edge* b, 
	      winged_edge_mesh<FaceRestriction>::edge* c)
	    {
		    this->cons_edges.push_back(a);
		    this->cons_edges.push_back(b);
		    this->cons_edges.push_back(c);
	    };
	    ~face() {};

	
	std::pair<fe_iterator, fe_iterator>
		get_edges() { return std::make_pair(this->cons_edges.begin(), this->cons_edges.end());}

	bool
		flip_normal()
	{
		return this->face_normal.flip();
	}
    };
    
template < typename FaceRestriction>
class winged_edge_mesh<FaceRestriction>::normal
{
	public:
		float x;
		float y;
		float z;

	public:
		normal()
		{
			this->x = 0.0f;
			this->y = 0.0f;
			this->z = 0.0f;
		}
		normal(float x, float y, float z)
		{
			this->x = x;
			this->y = y;
			this->z = z;
		}

		~normal()
		{
		}

		bool flip()
		{
			this->x = -1.0f * this->x;
			this->y = -1.0f * this->y;
			this->z = -1.0f * this->z;
		}
};

   
    template < typename FaceRestriction>
    class winged_edge_mesh<FaceRestriction>::vv_iterator : public std::iterator<std::input_iterator_tag, vertex_descriptor>
    {
    public:
	    winged_edge_mesh<FaceRestriction>::vertex* sender;
	    winged_edge_mesh<FaceRestriction>::vertex* p;
	    winged_edge_mesh<FaceRestriction>::edge_iterator e_i;
	    winged_edge_mesh<FaceRestriction>::edge_iterator edges_end;

    public:
	vv_iterator(const winged_edge_mesh<FaceRestriction>::edge_iterator edges_end)
	{
		this->e_i = edges_end;
		this->edges_end = edges_end;
		this->p = NULL;
		this->sender = NULL;
	}

	vv_iterator(const winged_edge_mesh<FaceRestriction>::vv_iterator& mit) : p(mit.p)
	{
		this->sender = mit.sender;
		this->p = mit.p;
		this->e_i = mit.e_i;
		this->edges_end = mit.edges_end;
	}

	vv_iterator(
	winged_edge_mesh<FaceRestriction>::edge_iterator ei, 
	winged_edge_mesh<FaceRestriction>::edge_iterator edges_end, 
	winged_edge_mesh<FaceRestriction>::vertex* sender, 
	winged_edge_mesh<FaceRestriction>::vertex* p)
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
	  winged_edge_mesh<FaceRestriction>::vertex* v1 = ((*e_i)->getVertices().first == this->sender) ?
			  (*e_i)->getVertices().second : (*e_i)->getVertices().first;
	  p = v1;
	  return *this;
      }
      bool operator==(const vv_iterator& rhs) {return e_i==rhs.e_i;}
      bool operator!=(const vv_iterator& rhs) {return e_i!=rhs.e_i;}
      winged_edge_mesh<FaceRestriction>::vertex_descriptor& operator*() {return p;}
    };
    
    template < typename FaceRestriction>
    winged_edge_mesh<FaceRestriction>::winged_edge_mesh()
    {
    }

    template < typename FaceRestriction>
    winged_edge_mesh<FaceRestriction>::~winged_edge_mesh()
    {
    }

    template < typename FaceRestriction>
    std::pair<typename winged_edge_mesh<FaceRestriction>::vertex_iterator,typename winged_edge_mesh<FaceRestriction>::vertex_iterator> 
    winged_edge_mesh<FaceRestriction>::getAllVertices()
    {
	    return std::make_pair(vertices.begin(), vertices.end());
    }
    
    template < typename FaceRestriction>
    typename winged_edge_mesh<FaceRestriction>::is_triangle winged_edge_mesh<FaceRestriction>::isTriangle()
    {
	    return isTriangleMesh;
    }

    template < typename FaceRestriction>
    bool 
    winged_edge_mesh<FaceRestriction>::add_vertex(const winged_edge_mesh<FaceRestriction>::vertex_descriptor v)
    {
	    if (std::find(vertices.begin(), vertices.end(), v) != vertices.end())
		    return false;
	    this->vertices.push_back(v);
	    return true;
    }
   
   
   template < typename FaceRestriction>
   typename winged_edge_mesh<FaceRestriction>::face_descriptor winged_edge_mesh<FaceRestriction>::create_face(
			  const winged_edge_mesh<FaceRestriction>::vertex_descriptor a,
			  const winged_edge_mesh<FaceRestriction>::vertex_descriptor b,
			  const winged_edge_mesh<FaceRestriction>::vertex_descriptor c
			  )
	  {
		  if (std::find(vertices.begin(), vertices.end(), a) == vertices.end())
			  return false;
		  if (std::find(vertices.begin(), vertices.end(), b) == vertices.end())
			  return false;
		  if (std::find(vertices.begin(), vertices.end(), c) == vertices.end())
			  return false;
		  winged_edge_mesh<FaceRestriction>::edge* ea = new winged_edge_mesh<FaceRestriction>::edge(a,b);
		  winged_edge_mesh<FaceRestriction>::edge* eb = new winged_edge_mesh<FaceRestriction>::edge(b,c);
		  winged_edge_mesh<FaceRestriction>::edge* ec = new winged_edge_mesh<FaceRestriction>::edge(c,a);


		  winged_edge_mesh<FaceRestriction>::face* f = new winged_edge_mesh<FaceRestriction>::face(ea, eb, ec);

		  ea->setFace(f);
		  eb->setFace(f);
		  ec->setFace(f);

		  this->edges.push_back(ea);
		  this->edges.push_back(eb);
		  this->edges.push_back(ec);

		  this->faces.push_back(f);

		  return f;
	  }
 
 
  template < typename FaceRestriction>
  bool winged_edge_mesh<FaceRestriction>::create_face(
			  const winged_edge_mesh<FaceRestriction>::edge_descriptor a,
			  const winged_edge_mesh<FaceRestriction>::edge_descriptor b,
			  const winged_edge_mesh<FaceRestriction>::edge_descriptor c)
  {
      if (std::find(edges.begin(), edges.end(), a) == edges.end())
	      return false;
      if (std::find(edges.begin(), edges.end(), b) == edges.end())
	      return false;
      if (std::find(edges.begin(), edges.end(), c) == edges.end())
	      return false;
      winged_edge_mesh<FaceRestriction>::face F(a,b,c);
      this->faces.push_back(F);
      return true;
  }
  
	  template < typename FaceRestriction>
  	  void winged_edge_mesh<FaceRestriction>::remove_face(const winged_edge_mesh<FaceRestriction>::face_descriptor f)
	  {
		  winged_edge_mesh<FaceRestriction>::face_iterator f_iter = std::find(faces.begin(), faces.end(), f);
		  if (f == faces.end())
			  return;
		  this->faces.erase(f_iter);
	  }

	  template < typename FaceRestriction>
	  std::pair<typename winged_edge_mesh<FaceRestriction>::vertex_iterator, typename winged_edge_mesh<FaceRestriction>::vertex_iterator> 
	  winged_edge_mesh<FaceRestriction>::get_all_vertices()
	  {
		  return std::make_pair(vertices.begin(), vertices.end());
	  }

	  template < typename FaceRestriction>
	  std::pair<typename winged_edge_mesh<FaceRestriction>::edge_iterator, typename winged_edge_mesh<FaceRestriction>::edge_iterator> 
	  winged_edge_mesh<FaceRestriction>::get_all_edges()
	  {
		  return std::make_pair(edges.begin(), edges.end());
	  }

	  template < typename FaceRestriction>
	  std::pair<typename winged_edge_mesh<FaceRestriction>::face_iterator, typename winged_edge_mesh<FaceRestriction>::face_iterator> 
	  winged_edge_mesh<FaceRestriction>::get_all_faces()
	  {
		  return std::make_pair(faces.begin(), faces.end());
	  }

	  template < typename FaceRestriction>
	  std::pair<typename winged_edge_mesh<FaceRestriction>::fe_iterator, typename winged_edge_mesh<FaceRestriction>::fe_iterator> 
	  winged_edge_mesh<FaceRestriction>::get_surrounding_edges(const face_descriptor f)
	  {
		  return f->get_edges();
	  }
  
  
  //==========BASIC CONCEPT=========

  template <typename Restriction>
  bool winged_edge_mesh_traits<Restriction>::add_vertex(
		  	  	  typename winged_edge_mesh<Restriction>::vertex_descriptor v,
		  	  	  class winged_edge_mesh<Restriction> *m)
  {
	  return  m->add_vertex(v);
  }

  template <typename Restriction>
  bool winged_edge_mesh_traits<Restriction>::create_face(
				  typename winged_edge_mesh<Restriction>::vertex_descriptor a,
				  typename winged_edge_mesh<Restriction>::vertex_descriptor b,
				  typename winged_edge_mesh<Restriction>::vertex_descriptor c,
		  	  	  class winged_edge_mesh<Restriction> *m)
  {
	  return  m->create_face(a,b,c);
  }

  template <typename Restriction>
  bool winged_edge_mesh_traits<Restriction>::remove_face(
				  typename winged_edge_mesh<Restriction>::face_descriptor f,
		  	  	  class winged_edge_mesh<Restriction> *m)
  {
	  return  m->remove_face(f);
  }

  template <typename Restriction>
  std::pair<typename winged_edge_mesh<Restriction>::vertex_iterator,
  	  	  	typename winged_edge_mesh<Restriction>::vertex_iterator>
  winged_edge_mesh_traits<Restriction>::get_all_vertices(const class winged_edge_mesh<Restriction>& m_)
  {
	  typedef winged_edge_mesh<Restriction> Mesh;
	  Mesh& m = const_cast<Mesh&>(m_);
	  return m.get_all_vertices();
  }

  template <typename Restriction>
  std::pair<typename winged_edge_mesh<Restriction>::edge_iterator,
  	  	  	typename winged_edge_mesh<Restriction>::edge_iterator>
  winged_edge_mesh_traits<Restriction>::get_all_edges(const class winged_edge_mesh<Restriction>& m_)
  {
	  typedef winged_edge_mesh<Restriction> Mesh;
	  Mesh& m = const_cast<Mesh&>(m_);
	  return m.get_all_edges();
  }

  template <typename Restriction>
  bool
  winged_edge_mesh_traits<Restriction>::is_isolated(
		  class winged_edge_mesh<Restriction>& m_,
		  class winged_edge_mesh<Restriction>::vertex* v)
  {
	  return v->is_isolated();
  }


  template <typename Restriction>
  std::pair<typename winged_edge_mesh<Restriction>::face_iterator,
  	  	  	typename winged_edge_mesh<Restriction>::face_iterator>
  winged_edge_mesh_traits<Restriction>::get_all_faces(class winged_edge_mesh<Restriction>& m_)
  {
	  typedef winged_edge_mesh<Restriction> Mesh;
	  Mesh& m = const_cast<Mesh&>(m_);
	  return m.get_all_faces();
  }

  template <typename Restriction>
  std::pair<typename winged_edge_mesh<Restriction>::vv_iterator,
  	  	  	typename winged_edge_mesh<Restriction>::vv_iterator>
	  winged_edge_mesh_traits<Restriction>::get_adjacent_vertices(
		  class winged_edge_mesh<Restriction>& m_,
		  class winged_edge_mesh<Restriction>::vertex* v)
  {
	  return v->get_adjacent_vertices();
  }
