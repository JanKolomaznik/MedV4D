/*
 * main.cpp
 *
 *  Created on: Jun 6, 2012
 *      Author: hmirap
 */

#include <OpenMesh/Core/Mesh/PolyMesh_ArrayKernelT.hh>
#include <OpenMesh/Core/IO/MeshIO.hh>

//#include "MedV4D/Imaging/ImageTools.h"


typedef OpenMesh::PolyMesh_ArrayKernelT<> MyMesh;

#include <iostream>
#include <deque>
#include <boost/graph/adjacency_matrix.hpp>
#include "meshes/winged_edge.h"

#include "algorithms/compute_components.h"
#include "algorithms/flip_normals.h"
#include "algorithms/triangulate.h"

#include "meshes/OpenMeshX.h"
#include "traits.h"

typedef winged_edge_mesh<triangleMesh> my_mesh;
typedef winged_edge_mesh_traits<triangleMesh> my_mesh_traits;

enum {A,B,C,D,N};

int main(int argc, char **argv)
{
	OpenMeshExtended mesh;

	typedef typename mesh_traits<OpenMeshExtended>::vertex_descriptor vd;

	//MyMesh mesh;

	OpenMeshXTraits::vertex_descriptor vhandle[8];

	vhandle[0] = mesh.add_vertex(OpenMeshExtended::Point(-1, -1,  1));
	vhandle[1] = mesh.add_vertex(OpenMeshExtended::Point( 1, -1,  1));
	vhandle[2] = mesh.add_vertex(OpenMeshExtended::Point( 1,  1,  1));
	vhandle[3] = mesh.add_vertex(OpenMeshExtended::Point(-1,  1,  1));
	vhandle[4] = mesh.add_vertex(OpenMeshExtended::Point(-1, -1, -1));
	vhandle[5] = mesh.add_vertex(OpenMeshExtended::Point( 1, -1, -1));
	vhandle[6] = mesh.add_vertex(OpenMeshExtended::Point( 1,  1, -1));
	vhandle[7] = mesh.add_vertex(OpenMeshExtended::Point(-1,  1, -1));


/*  testujem iteratory */

auto test_vv_iter_pair = OpenMeshXTraits::get_adjacent_vertices(mesh, vhandle[0]);
std::cout << "pre izolovany vrchol vyhodi rovny iterator:" << (test_vv_iter_pair.first == test_vv_iter_pair.second) << " great!" << std::endl;

/*# testujem iteratory*/



	std::vector<OpenMeshXTraits::vertex_descriptor>  face_vhandles;

	face_vhandles.clear();
	face_vhandles.push_back(vhandle[0]);
	face_vhandles.push_back(vhandle[1]);
	face_vhandles.push_back(vhandle[2]);
	face_vhandles.push_back(vhandle[3]);


	
	
	/*0 0 1*/
/*	
	face_vhandles.clear();
	face_vhandles.push_back(vhandle[3]);
	face_vhandles.push_back(vhandle[2]);
	face_vhandles.push_back(vhandle[1]);
	face_vhandles.push_back(vhandle[0]);
*/
	/*0 0 -1*/
	OpenMesh::PolyMesh_ArrayKernelT<>::FaceHandle omx_face = mesh.add_face(face_vhandles);
	
	OpenMeshExtended::Normal normal = mesh.calc_face_normal(omx_face);
	std::cout << "normala:" << normal << std::endl;

	flip_normals<OpenMeshExtended, advanced_mesh_traits<OpenMeshExtended>>(mesh);	


	auto old_face_pair = mesh_traits<OpenMeshExtended>::get_all_faces(mesh);
	for (auto i = old_face_pair.first; i != old_face_pair.second; ++i)
	{
		auto old_face = *i;
		OpenMeshExtended::Normal n = mesh.calc_face_normal(old_face);
		std::cout << "norm:" << n << std::endl;
	} 

	triangulate<OpenMeshExtended, advanced_mesh_traits<OpenMeshExtended>>(mesh);
	
	std::cout << "pocet je " << compute_components<OpenMeshExtended, OpenMeshXTraits>(mesh) << std::endl;


	OpenMeshXTraits::face_descriptor fh = *mesh.faces_begin();


	for ( auto fvi = mesh.fv_begin(fh); fvi < mesh.fv_end(fh); ++fvi) {
		std::cout << "h" << std::endl;
	}

	for (auto e_it = mesh.edges_begin(); e_it != mesh.edges_end(); ++e_it)
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

/* 2 testujem iteratory */

auto test_vv_iter_pair2 = my_mesh_traits::get_adjacent_vertices(G, &v1);
std::cout << "2: pre izolovany vrchol vyhodi rovny iterator:" << (test_vv_iter_pair2.first == test_vv_iter_pair2.second) << " great!" << std::endl;

/*# 2 testujem iteratory*/




	my_mesh_traits::add_vertex(&v1, &G);
	my_mesh_traits::add_vertex(&v2, &G);
	my_mesh_traits::add_vertex(&v3, &G);
	my_mesh_traits::add_vertex(&v4, &G);
	my_mesh_traits::add_vertex(&v5, &G);
	my_mesh_traits::add_vertex(&v6, &G);
	my_mesh_traits::add_vertex(&v7, &G);
	my_mesh_traits::add_vertex(&v8, &G);

	my_mesh_traits::create_face(&v1, &v2, &v3, &G);
	my_mesh_traits::create_face(&v2, &v3, &v4, &G);
	my_mesh_traits::create_face(&v6, &v7, &v8, &G);

	std::cout << std::endl;

	auto my_pair = my_mesh_traits::get_all_vertices(G);
	for (auto i = my_pair.first; i != my_pair.second; ++i) {
		std::cout << (*i)->get_id() << ", ";
	}
	std::cout << std::endl;

	auto my_pair_edges = my_mesh_traits::get_all_edges(G);
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


	//M4D::Imaging::AImage::Ptr image = M4D::Imaging::ImageFactory::LoadDumpedImage( path );

	return 0;
}
