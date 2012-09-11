#ifdef OPENMESHX_H_
#define OPENMESHX_H_

std::pair<OpenMeshXTraits::fv_iterator, OpenMeshXTraits::fv_iterator>
OpenMeshX::get_surrounding_vertices(OpenMeshXTraits::face_descriptor fd)
{
    return std::make_pair(this->fv_begin(fd), this->fv_end(fd));
}

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

#endif