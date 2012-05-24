#ifndef GRAPH_TOOLS_H
#define GRAPH_TOOLS_H

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/shared_ptr.hpp>

#include "MedV4D/Common/Types.h"

typedef boost::adjacency_list<
		boost::vecS, 
		boost::vecS, 
		boost::undirectedS,
		boost::no_property, 
		boost::property<boost::edge_weight_t, float> 
		> WeightedUndirectedGraph;

typedef boost::graph_traits < WeightedUndirectedGraph > WeightedUndirectedGraphTraits;

typedef boost::adjacency_list<
		boost::vecS, 
		boost::vecS, 
		boost::bidirectionalS,
		boost::no_property, 
		boost::property<boost::edge_weight_t, float> 
		> WeightedBidirectionalGraph;

typedef boost::graph_traits < WeightedBidirectionalGraph > WeightedBidirectionalGraphTraits;
		

struct WeightedEdgeListGraph
{
	typedef boost::shared_ptr< WeightedEdgeListGraph > Ptr;
	struct EdgeRecord
	{
		EdgeRecord( uint32 aFirst, uint32 aSecond )
		{
			first = std::min( aFirst, aSecond );
			second = std::max( aFirst, aSecond );
		}

		EdgeRecord(): second(0), first(0)
		{ }
		uint32 second;
		uint32 first;
	};
	
	WeightedEdgeListGraph(): mVertexCount( 0 ), mEdgeCount( 0 )
	{}
	
	
	std::vector< EdgeRecord > mEdges;
	std::vector< float > mWeights;
	size_t mVertexCount;
	size_t mEdgeCount;
};



void
computeMinCut( WeightedUndirectedGraph & aGraph );

void
computeMaxFlow( WeightedBidirectionalGraph & aGraph );


#endif //GRAPH_TOOLS_H