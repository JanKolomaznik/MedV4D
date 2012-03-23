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

void
computeMinCut( WeightedUndirectedGraph & aGraph );


#endif //GRAPH_TOOLS_H