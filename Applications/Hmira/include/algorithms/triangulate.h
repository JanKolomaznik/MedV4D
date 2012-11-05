#include <iostream>

template <typename TMesh, typename TMesh_Traits = mesh_traits<TMesh>>
int triangulate(TMesh& m)
{
	typedef typename TMesh_Traits::face_descriptor face_descriptor;
        typedef typename TMesh_Traits::face_iterator face_iterator;
	typedef typename TMesh_Traits::vertex_descriptor vertex_descriptor;
	typedef typename TMesh_Traits::fv_iterator fv_iterator;

        auto all_faces = TMesh_Traits::get_all_faces(m);

	for (auto i = all_faces.first; i != all_faces.second; ++i)
	{
		//std::cout << "a" << std::endl;
		face_descriptor fd = *i;
		auto surrounding_vertices = TMesh_Traits::get_surrounding_vertices( m, fd);
		for (auto fvi = surrounding_vertices.first; fvi != surrounding_vertices.second; ++i)
		{
		}
	}

	return 0;
}
