#ifdef OPENMESHX_H_
#define OPENMESHX_H_


std::pair<OpenMeshXTraits::fv_iterator, OpenMeshXTraits::fv_iterator>
OpenMeshXTraits::get_surrounding_vertices(const OpenMeshExtended& m_, OpenMeshXTraits::face_descriptor fd)
{
	typedef OpenMeshExtended Mesh;
	Mesh& m = const_cast<Mesh&>(m_);
	return std::make_pair(m.fv_begin(fd), m.fv_end(fd));
}

//=================CONCEPTS======================

bool OpenMeshXTraits::remove_vertex(
					  OpenMeshXTraits::vertex_descriptor v,
		  	  	  OpenMeshExtended &m)
{
	 m.delete_vertex(v);
	 return true;
}

bool OpenMeshXTraits::create_face(
				  typename OpenMeshXTraits::vertex_descriptor a,
				  typename OpenMeshXTraits::vertex_descriptor b,
				  typename OpenMeshXTraits::vertex_descriptor c,
		  	  	  OpenMeshExtended *m)
{
	std::vector<OpenMeshXTraits::vertex_descriptor>  face_vhandles;

	face_vhandles.clear();
	face_vhandles.push_back(a);
	face_vhandles.push_back(b);
	face_vhandles.push_back(c);
	m->add_face(face_vhandles);
	return true;
}

bool OpenMeshXTraits::remove_face(
				  typename OpenMeshXTraits::face_descriptor f,
		  	  	  OpenMeshExtended *m)
{
	  m->delete_face(f);
	  return true;
}

std::pair<typename OpenMeshXTraits::vertex_iterator,
	  	  	typename OpenMeshXTraits::vertex_iterator>
OpenMeshXTraits::get_all_vertices(const OpenMeshExtended& m_)
{
	  typedef OpenMeshExtended Mesh;
	  Mesh& m = const_cast<Mesh&>(m_);
	  return std::make_pair(m.vertices_begin(), m.vertices_end());
}

std::pair<typename OpenMeshXTraits::edge_iterator,
	  	  	typename OpenMeshXTraits::edge_iterator>
OpenMeshXTraits::get_all_edges(const OpenMeshExtended& m_)
{
	  typedef OpenMeshExtended Mesh;
	  Mesh& m = const_cast<Mesh&>(m_);
	  return std::make_pair(m.edges_begin(), m.edges_end());
}


std::pair<typename OpenMeshXTraits::face_iterator,
	  	  	typename OpenMeshXTraits::face_iterator>
OpenMeshXTraits::get_all_faces(const OpenMeshExtended& m_)
{
	  typedef OpenMeshExtended Mesh;
	  Mesh& m = const_cast<Mesh&>(m_);
	  return std::make_pair(m.faces_begin(), m.faces_end());
}

//=========== VERTEX ADJACENCY CONCEPT ===========

bool OpenMeshXTraits::is_isolated(const OpenMeshExtended& m_,
		OpenMeshXTraits::vertex_descriptor v)
{
	  typedef OpenMeshExtended Mesh;
	  Mesh& m = const_cast<Mesh&>(m_);
	  return m.is_isolated(v);
}

std::pair<typename OpenMeshXTraits::vv_iterator,
	  	  	typename OpenMeshXTraits::vv_iterator>
OpenMeshXTraits::get_adjacent_vertices(
		const OpenMeshExtended& m_,
		  OpenMeshXTraits::vertex_descriptor v)
		  {
	  typedef OpenMeshExtended Mesh;
	  Mesh& m = const_cast<Mesh&>(m_);
	  return std::make_pair(m.vv_begin(v), m.vv_end(v));
		  }

std::pair<typename OpenMeshXTraits::ve_iterator,typename OpenMeshXTraits::ve_iterator>
OpenMeshXTraits::get_adjacent_edges(
		const OpenMeshExtended& m_,
		OpenMeshXTraits::vertex_descriptor v)
		  {

	  typedef OpenMeshExtended Mesh;
	  Mesh& m = const_cast<Mesh&>(m_);
	  return std::make_pair(m.ve_begin(v),m.ve_end(v));
		  }

advanced_mesh_traits<OpenMeshExtended>::normal
advanced_mesh_traits<OpenMeshExtended>::get_face_normal(
	const OpenMeshExtended& m_,
	OpenMeshXTraits::face_descriptor f)
	{
		typedef OpenMeshExtended Mesh;
		Mesh& m = const_cast<Mesh&>(m_);
		return m.calc_face_normal(f);
	}

bool
advanced_mesh_traits<OpenMeshExtended>::flip_face_normal(
	OpenMeshExtended& m_,
	face_descriptor& f)
	{
		std::vector<OpenMeshExtended::VertexHandle> face_vhandles;
		std::deque<OpenMeshExtended::VertexHandle> vdx_vector;
		
		typedef OpenMeshExtended Mesh;
		Mesh& m = const_cast<Mesh&>(m_);

		auto fv1 = m.fv_begin(f);
		auto fv2 = m.fv_end(f);

        	for (auto vdx = fv1; vdx != fv2; ++vdx)
        	{
                	vdx_vector.push_front(vdx.handle());
        	}

		face_vhandles.clear();        

		for (auto i : vdx_vector)
        	{
                	face_vhandles.push_back(i);
        	}

        	m.delete_face(f, false);
		m.garbage_collection();

		f = m.add_face(face_vhandles);
		return true;
	}

#endif
