#include "GraphCutSegmentation.hpp"

#include "GridGraph_3D_6C.h"
#include "GridGraph_3D_6C_MT.h"

#include <array>
#include <limits>
#include <memory>


template<typename TImage, typename TMask, typename TGraph>
void
graphCutSegmentation(const TImage &aImage, const TMask &aMarkerData, TMask &aSegmentationMask)
{
	TGraph graph;
	buildGraph(graph, aImage, aMarkerData);

	connectSourceAndSink(graph, aMarkerData);

	findMinCut(graph);

	constructSegmentationMask(graph, aSegmentationMask);
}

typedef GridGraph_3D_6C<int64_t, int64_t, int64_t> GridCutGraphST;
typedef GridGraph_3D_6C_MT<int64_t, int64_t, int64_t> GridCutGraphMT;

template<typename TGraphImplementation>
struct GridCutGraph
{
	typedef int64_t TerminalCapacity;
	typedef int64_t NeighborCapacity;
	typedef TGraphImplementation Graph;

	std::unique_ptr<Graph> graph;
};

template<typename TGraphImplementation, typename TImage, typename TMask>
void
buildGraph(GridCutGraph<TGraphImplementation> &aGraph, const TImage &aImage, const TMask &aMarkerData)
{
	typedef typename TImage::PointType Coords;
	typedef GridCutGraph<TGraphImplementation> GridCutGraphWrapper;

	auto imageSize = aImage.GetSize();
	aGraph.graph = std::unique_ptr<typename GridCutGraphWrapper::Graph>(new typename GridCutGraphWrapper::Graph(imageSize[0], imageSize[1], imageSize[2]/*, 8, 1000*/));

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
				auto currentValue = aImage.GetElement(nodeCoords);
				/*auto marker = aMarkerData.GetElement(nodeCoords);
				switch (marker) {
				case 255:
					aGraph.graph->set_terminal_cap(node, 0, 100000);
					break;
				case 128:
					aGraph.graph->set_terminal_cap(node, 100000, 0);
					break;
				default:
					break;
				}*/
				for (int i = 0; i < 3; ++i) {
					Coords neighborCoords1 = nodeCoords;
					Coords neighborCoords2 = nodeCoords;
					for (int j = 0; j < 3; ++j) {
						neighborCoords1[j] += cOffsets[2*i][j];
						neighborCoords2[j] += cOffsets[2*i + 1][j];
					}
					if (nodeCoords[i] > 0) {
						auto neighborValue = aImage.GetElement(neighborCoords1);
						int weight = currentValue - neighborValue;
						weight = std::exp(-(weight * weight) / 0.1);
						aGraph.graph->set_neighbor_cap(
								node,
								cOffsets[2*i][0],
								cOffsets[2*i][1],
								cOffsets[2*i][2],
								weight);
					}
					if (nodeCoords[i] < int(imageSize[i]) - 1) {
						auto neighborValue = aImage.GetElement(neighborCoords2);
						int weight = currentValue - neighborValue;
						weight = std::exp(-(weight * weight) / 200);
						aGraph.graph->set_neighbor_cap(
								node,
								cOffsets[2*i + 1][0],
								cOffsets[2*i + 1][1],
								cOffsets[2*i + 1][2],
								weight);
					}
				}
			}
		}
	}
}

template<typename TGraphImplementation, typename TMask>
void
connectSourceAndSink(GridCutGraph<TGraphImplementation> &aGraph, const TMask &aMarkerData)
{
	typedef typename TMask::PointType Coords;
	typedef typename TMask::SizeType Size;

	auto imageSize = aMarkerData.GetSize();
	for (int z = 0; z < int(imageSize[2]); ++z) {
		for (int y = 0; y < int(imageSize[1]); ++y) {
			for(int x = 0; x < int(imageSize[0]); ++x) {
				const int node = aGraph.graph->node_id(x,y,z);
				Coords nodeCoords(x, y, z);
				auto marker = aMarkerData.GetElement(nodeCoords);
				switch (marker) {
				case 255:
					aGraph.graph->set_terminal_cap(node, 0, 100000);
					break;
				case 128:
					aGraph.graph->set_terminal_cap(node, 100000, 0);
					break;
				default:
					aGraph.graph->set_terminal_cap(node, 0, 0);
					break;
				}
			}
		}
	}
}

template<typename TGraphImplementation>
void
findMinCut(GridCutGraph<TGraphImplementation> &aGraph)
{
	aGraph.graph->compute_maxflow();
}

template<typename TGraphImplementation, typename TMask>
void
constructSegmentationMask(GridCutGraph<TGraphImplementation> &aGraph, TMask &aSegmentationMask)
{
	typedef typename TMask::PointType Coords;

	auto imageSize = aSegmentationMask.GetSize();

	for (int z = 0; z < int(imageSize[2]); ++z) {
		for (int y = 0; y < int(imageSize[1]); ++y) {
			for(int x = 0; x < int(imageSize[0]); ++x) {
				const int node = aGraph.graph->node_id(x,y,z);
				Coords nodeCoords(x, y, z);
				auto segment = aGraph.graph->get_segment(node);
				aSegmentationMask.GetElement(nodeCoords) = segment ? 255 : /*128*/0;
			}
		}
	}
}

void computeGraphCutSegmentation(const M4D::Imaging::AImageDim<3> &aImage, const M4D::Imaging::Mask3D &aMarkerData, M4D::Imaging::Mask3D &aSegmentationMask)
{
	NUMERIC_TYPE_TEMPLATE_SWITCH_MACRO(aImage.GetElementTypeID(),
		typedef const M4D::Imaging::Image<TTYPE, 3> ConstImageType;
		ConstImageType &image = static_cast<ConstImageType &>(aImage);
		graphCutSegmentation<ConstImageType, M4D::Imaging::Mask3D, GridCutGraph<GridCutGraphST>>(image, aMarkerData, aSegmentationMask);
	);
}
