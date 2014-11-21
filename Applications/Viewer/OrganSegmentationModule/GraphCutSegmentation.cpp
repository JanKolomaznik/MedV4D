#include "GraphCutSegmentation.hpp"

#include "GridGraph_3D_6C.h"
//#include "GridGraph_3D_6C_MT.h"

#include <array>
#include <limits>
#include <memory>


template<typename TImage, typename TMask, typename TGraph>
void
graphCutSegmentation(const TImage &aImage, const TMask &aMarkerData, TMask &aSegmentationMask)
{
	TGraph graph;
	buildGraph(graph, aImage);

	/*connectSourceAndSink(graph, aMarkerData);

	findMinCut(graph);

	constructSegmentationMask(graph, aSegmentationMask);*/
}

struct GridCutGraph
{
	typedef int64_t TerminalCapacity;
	typedef int64_t NeighborCapacity;
	typedef GridGraph_3D_6C<TerminalCapacity, NeighborCapacity, int64_t> Graph;

	std::unique_ptr<Graph> graph;
};

template<typename TImage>
void
buildGraph(GridCutGraph &aGraph, const TImage &aImage)
{
	typedef typename TImage::PointType Coords;
	typedef typename TImage::SizeType Size;

	auto imageSize = aImage.GetSize();
	//template<typename type_terminal_cap,typename type_neighbor_cap>
	/*const int w = mfi->width;
	const int h = mfi->height;
	const int d = mfi->depth;*/

	/*const type_terminal_cap* cap_source = (type_terminal_cap*)mfi->cap_source;
	const type_terminal_cap* cap_sink   = (type_terminal_cap*)mfi->cap_sink;*/

/*	const type_neighbor_cap* cap_neighbor[6] = { (type_neighbor_cap*)(mfi->cap_neighbor[0]),
						     (type_neighbor_cap*)(mfi->cap_neighbor[1]),
						     (type_neighbor_cap*)(mfi->cap_neighbor[2]),
						     (type_neighbor_cap*)(mfi->cap_neighbor[3]),
						     (type_neighbor_cap*)(mfi->cap_neighbor[4]),
						     (type_neighbor_cap*)(mfi->cap_neighbor[5]) };*/


	aGraph.graph = std::unique_ptr<GridCutGraph::Graph>(new GridCutGraph::Graph(imageSize[0], imageSize[1], imageSize[2]));

	static const std::array<std::array<int, 3>, 6> cOffsets = {
		std::array<int, 3>{{-1, 0, 0}},
		std::array<int, 3>{{+1, 0, 0}},
		std::array<int, 3>{{ 0,-1, 0}},
		std::array<int, 3>{{ 0,+1, 0}},
		std::array<int, 3>{{ 0, 0,-1}},
		std::array<int, 3>{{ 0, 0,+1}}
	};

	for (int z = 0; z < int(imageSize[2]); ++z) {
		for (int y = 0; y < int(imageSize[1]); ++y) {
			for(int x = 0; x < int(imageSize[0]); ++x) {
				const int node = aGraph.graph->node_id(x,y,z);
				Coords nodeCoords(x, y, z);
				//aGraph.graph->set_terminal_cap(node, cap_source[x+y*w+z*(w*h)],cap_sink[x+y*w+z*(w*h)]);

				for (int i = 0; i < 3; ++i) {
					if (nodeCoords[i] > 0) {
						int weight = 0;
						aGraph.graph->set_neighbor_cap(
								node,
								cOffsets[2*i][0],
								cOffsets[2*i][1],
								cOffsets[2*i][2],
								weight);
					}
					if (nodeCoords[i] < imageSize[i] - 1) {
						int weight = 0;
						aGraph.graph->set_neighbor_cap(
								node,
								cOffsets[2*i + 1][0],
								cOffsets[2*i + 1][1],
								cOffsets[2*i + 1][2],
								weight);
					}
				}
/*
				if (x > 0  ) {
					aGraph.graph->set_neighbor_cap(node,-1, 0, 0,cap_neighbor[MFI::ARC_LEE][x+y*w+z*(w*h)]);
				}
				if (x < w-1) {
					aGraph.graph->set_neighbor_cap(node,+1, 0, 0,cap_neighbor[MFI::ARC_GEE][x+y*w+z*(w*h)]);
				}
				if (y > 0  ) {
					aGraph.graph->set_neighbor_cap(node, 0,-1, 0,cap_neighbor[MFI::ARC_ELE][x+y*w+z*(w*h)]);
				}
				if (y < h-1) {
					aGraph.graph->set_neighbor_cap(node, 0,+1, 0,cap_neighbor[MFI::ARC_EGE][x+y*w+z*(w*h)]);
				}
				if (z > 0  ) {
					aGraph.graph->set_neighbor_cap(node, 0, 0,-1,cap_neighbor[MFI::ARC_EEL][x+y*w+z*(w*h)]);
				}
				if (z < d-1) {
					aGraph.graph->set_neighbor_cap(node, 0, 0,+1,cap_neighbor[MFI::ARC_EEG][x+y*w+z*(w*h)]);
				}*/
			}
		}
	}
//	CLOCK_STOP(time_init);

//	CLOCK_START();
	aGraph.graph->compute_maxflow();
//	CLOCK_STOP(time_maxflow)

	/*CLOCK_START();
	*out_maxflow = graph->get_flow();

	for(int z=0;z<d;z++)
	for(int y=0;y<h;y++)
	for(int x=0;x<w;x++)
	{
	  out_label[x+y*w+z*(w*h)] = graph->get_segment(graph->node_id(x,y,z));
	}

	delete graph;*/

}


void computeGraphCutSegmentation(const M4D::Imaging::AImageDim<3> &aImage, const M4D::Imaging::Mask3D &aMarkerData, M4D::Imaging::Mask3D &aSegmentationMask)
{
	NUMERIC_TYPE_TEMPLATE_SWITCH_MACRO(aImage.GetElementTypeID(),
		typedef const M4D::Imaging::Image<TTYPE, 3> ConstImageType;
		ConstImageType &image = static_cast<ConstImageType &>(aImage);
		graphCutSegmentation<ConstImageType, M4D::Imaging::Mask3D, GridCutGraph>(image, aMarkerData, aSegmentationMask);
	);
}
