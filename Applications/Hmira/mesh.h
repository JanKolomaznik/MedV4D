/*
 * mesh.h
 *
 *  Created on: Jun 6, 2012
 *      Author: hmirap
 */

#ifndef MESH_H_
#define MESH_H_

#include <vector>

template <class Traits>
class mesh
{
public:
	mesh();
	virtual ~mesh();

	std::vector<int> VertexList;
	std::vector<int> FaceList;
};

#endif /* MESH_H_ */
