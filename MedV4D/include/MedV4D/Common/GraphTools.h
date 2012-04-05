#ifndef GRAPH_TOOLS_H
#define GRAPH_TOOLS_H

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>

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
		
void
computeMinCut( WeightedUndirectedGraph & aGraph );

void
computeMaxFlow( WeightedBidirectionalGraph & aGraph );


#endif //GRAPH_TOOLS_H