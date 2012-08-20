/*
 * main.cpp
 *
 *  Created on: Jun 6, 2012
 *      Author: hmirap
 */

#include <OpenMesh/Core/Mesh/PolyMesh_ArrayKernelT.hh>
#include <OpenMesh/Core/IO/MeshIO.hh>

typedef OpenMesh::PolyMesh_ArrayKernelT<> MyMesh;

#include <iostream>
#include "winged_edge.h"
#include "compute_components.h"
#include <boost/graph/adjacency_matrix.hpp>

#include "OpenMeshX.h"
#include "traits.h"

typedef winged_edge_mesh<triangleMesh> my_mesh;
typedef winged_edge_mesh_traits<triangleMesh> my_mesh_traits;

enum {A,B,C,D,N};

int main(int argc, char **argv)
{
	OpenMeshX mesh;

	typedef typename mesh_traits<OpenMeshX>::vertex_descriptor vd;

	//MyMesh mesh;

	OpenMeshX::vertex_descriptor vhandle[8];

	vhandle[0] = mesh.add_vertex(OpenMeshX::Point(-1, -1,  1));
	vhandle[1] = mesh.add_vertex(OpenMeshX::Point( 1, -1,  1));
	vhandle[2] = mesh.add_vertex(OpenMeshX::Point( 1,  1,  1));
	vhandle[3] = mesh.add_vertex(OpenMeshX::Point(-1,  1,  1));
	vhandle[4] = mesh.add_vertex(OpenMeshX::Point(-1, -1, -1));
	vhandle[5] = mesh.add_vertex(OpenMeshX::Point( 1, -1, -1));
	vhandle[6] = mesh.add_vertex(OpenMeshX::Point( 1,  1, -1));
	vhandle[7] = mesh.add_vertex(OpenMeshX::Point(-1,  1, -1));


	std::vector<OpenMeshX::vertex_descriptor>  face_vhandles;

	face_vhandles.clear();
	face_vhandles.push_back(vhandle[0]);
	face_vhandles.push_back(vhandle[1]);
	face_vhandles.push_back(vhandle[2]);
	face_vhandles.push_back(vhandle[3]);
	mesh.add_face(face_vhandles);


	std::cout << "pocet je " << compute_components<OpenMeshX, OpenMeshXTraits>(mesh) << std::endl;


	OpenMeshX::face_descriptor fh = *mesh.faces_begin();


	for ( OpenMeshX::fv_iterator fvi = mesh.fv_begin(fh); fvi < mesh.fv_end(fh); ++fvi) {
		std::cout << "h" << std::endl;
	}

	for (OpenMeshX::edge_iterator e_it = mesh.edges_begin(); e_it != mesh.edges_end(); ++e_it)
	{
		std::cout << "joe" << e_it->idx() << std::endl;
	}

	boost::adjacency_matrix<boost::directedS> a(N);
	add_edge(A, B, a);

	my_mesh G;

	my_mesh::vertex v1(1);
	my_mesh::vertex v2(2);
	my_mesh::vertex v3(3);
	my_mesh::vertex v4(4);
	my_mesh::vertex v5(5);
	my_mesh::vertex v6(6);
	my_mesh::vertex v7(7);
	my_mesh::vertex v8(8);

	add_vertex(&v1, &G);
	add_vertex(&v2, &G);
	add_vertex(&v3, &G);
	add_vertex(&v4, &G);
	add_vertex(&v5, &G);
	add_vertex(&v6, &G);
	add_vertex(&v7, &G);
	add_vertex(&v8, &G);

	create_face(&v1, &v2, &v3, &G);
	create_face(&v2, &v3, &v4, &G);
	create_face(&v6, &v7, &v8, &G);

	std::cout << std::endl;

	auto my_pair = get_all_vertices(G);
	for (auto i = my_pair.first; i != my_pair.second; ++i) {
		std::cout << (*i)->get_id() << ", ";
	}
	std::cout << std::endl;

	auto my_pair_edges = get_all_edges(G);
	for (auto i = my_pair_edges.first; i != my_pair_edges.second; ++i) {
		std::cout << ((*i)->getVertices().first ? (*i)->getVertices().first->get_id() : 0) << "-" ;
		std::cout << ((*i)->getVertices().second ? (*i)->getVertices().second->get_id() : 0) << ",";
	}
	std::cout << std::endl;


	std::cout << G.isTriangle() << std::endl;
	std::cout << "joe more!!!!";

	auto my_pair_vv = v1.get_adjacent_vertices();// get_adjacent_vertices(G, &v1);
	for (auto i = my_pair_vv.first; i != my_pair_vv.second; ++i) {
		std::cout << (*i)->get_id() << ",";
	}
	std::cout << std::endl;

	std::cout << "pocet je: " << compute_components<my_mesh, my_mesh_traits>(G) << std::endl;

	std::cout << "norma: c++0x" << std::endl;

	return 0;
}
