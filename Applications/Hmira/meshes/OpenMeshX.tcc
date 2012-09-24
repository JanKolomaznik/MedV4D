#ifdef OPENMESHX_H_
#define OPENMESHX_H_

class mesh_traits<OpenMeshExtended>::my_ve_iterator : public OpenMeshExtended::VEIter
{
	public: 
                typedef OpenMesh::PolyConnectivity& mesh_ref;

                my_ve_iterator() : 
                        OpenMesh::Iterators::VertexEdgeIterT< OpenMesh::PolyConnectivity >() {};
                
                my_ve_iterator (mesh_ref _mesh, OpenMesh::PolyConnectivity::VertexHandle _start, bool _end) :
                        OpenMesh::Iterators::VertexEdgeIterT< OpenMesh::PolyConnectivity >(_mesh, _start, _end) {};

                my_ve_iterator(mesh_ref _mesh, OpenMesh::PolyConnectivity::HalfedgeHandle _heh, bool _end) : 
                        OpenMesh::Iterators::VertexEdgeIterT< OpenMesh::PolyConnectivity >(_mesh, _heh, _end) {};

                my_ve_iterator(const OpenMesh::Iterators::VertexEdgeIterT< OpenMesh::PolyConnectivity >& _rhs) :
                        OpenMesh::Iterators::VertexEdgeIterT< OpenMesh::PolyConnectivity >(_rhs) {};

                bool operator==(const my_ve_iterator& _rhs) const 
                {
                        return 
                        ((mesh_   == _rhs.mesh_) &&
                        (start_  == _rhs.start_) &&
                        (heh_    == _rhs.heh_));
                }
 
 
                bool operator!=(const my_ve_iterator& _rhs) const
                {
                        return !operator==(_rhs);
                }

                edge_descriptor operator*() { return this->handle(); }
                edge_descriptor operator->() { return this->handle(); }


};
                 

class mesh_traits<OpenMeshExtended>::my_fv_iterator : public OpenMesh::Iterators::FaceVertexIterT< OpenMesh::PolyConnectivity >
{
        public:
                typedef OpenMesh::PolyConnectivity& mesh_ref;

/*      TODO prekonzultovať 
*       constructor inheritance v norme c++0x 
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

                vertex_descriptor operator*() { return this->handle(); }
                vertex_descriptor operator->() { return this->handle(); }
};
             

class mesh_traits<OpenMeshExtended>::my_vv_iterator : public OpenMesh::Iterators::VertexVertexIterT< OpenMesh::PolyConnectivity >
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

                vertex_descriptor operator*() { return this->handle(); }
                vertex_descriptor operator->() { return this->handle(); }

};


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
		  	  	  OpenMeshExtended& m)
{
	std::vector<OpenMeshXTraits::vertex_descriptor>  face_vhandles;

	face_vhandles.clear();
	face_vhandles.push_back(a);
	face_vhandles.push_back(b);
	face_vhandles.push_back(c);
	m.add_face(face_vhandles);
	return true;
}

bool OpenMeshXTraits::remove_face(
				  typename OpenMeshXTraits::face_descriptor f,
		  	  	  OpenMeshExtended &m)
{
	  m.delete_face(f);
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
