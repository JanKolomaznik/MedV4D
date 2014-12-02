#include "MedV4D/Common/GraphTools.h"
#include "MedV4D/Common/Debug.h"
#include "MedV4D/Common/Log.h"
#include "MedV4D/Common/TimeMeasurement.h"


#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/one_bit_color_map.hpp>
#include <boost/graph/stoer_wagner_min_cut.hpp>
#include <boost/graph/boykov_kolmogorov_max_flow.hpp>
#include <boost/property_map/property_map.hpp>
#include <boost/typeof/typeof.hpp>
#include <boost/graph/bipartite.hpp>
#include <boost/graph/connected_components.hpp>
#include <vector>

struct edge_t
{
  unsigned long first;
  unsigned long second;
};

void
computeMinCut( WeightedUndirectedGraph & aGraph )
{
	M4D::Common::Clock clock;

	//size_t comp = boost::is_bipartite( aGraph );
	std::vector<int> component(boost::num_vertices(aGraph));
	size_t comp = boost::connected_components(aGraph, &component[0]);

	LOG( "Number of components = " << comp );

	BOOST_AUTO( parities, boost::make_one_bit_color_map(num_vertices(aGraph), get(boost::vertex_index, aGraph)) );



	/*edge_t edges[] = {{3, 4}, {3, 6}, {3, 5}, {0, 4}, {0, 1}, {0, 6}, {0, 7},
	{0, 5}, {0, 2}, {4, 1}, {1, 6}, {1, 5}, {6, 7}, {7, 5}, {5, 2}, {3, 4}};

	// for each of the 16 edges, define the associated edge weight. ws[i] is the weight for the edge
	// that is described by edges[i].
	float ws[] = {0, 3, 1, 3, 1, 2, 6, 1, 8, 1, 1, 80, 2, 1, 1, 4};

	aGraph = WeightedUndirectedGraph(edges, edges + 16, ws, 8, 16);*/

	size_t i;
	for (i = 0; i < boost::num_vertices(aGraph); ++i) {
		if (boost::out_degree( i, aGraph ) == 0) {
			std::cout << i+1 << " - degree " << boost::out_degree( i, aGraph ) << std::endl;
		}
	}



	float w = boost::stoer_wagner_min_cut(aGraph, get(boost::edge_weight, aGraph), boost::parity_map(parities));
	LOG( "min cut weight = " << w );


	//std::pair<WeightedUndirectedGraph::edge_iterator, WeightedUndirectedGraph::edge_iterator> es = boost::edges(aGraph);

	WeightedUndirectedGraphTraits::edge_iterator ei, ei_end;
	for (boost::tie(ei, ei_end) = boost::edges(aGraph); ei != ei_end; ++ei) {
		WeightedUndirectedGraphTraits::edge_descriptor e = *ei;
		WeightedUndirectedGraphTraits::vertex_descriptor u = boost::source(e, aGraph);
		WeightedUndirectedGraphTraits::vertex_descriptor v = boost::target(e, aGraph);
		if(
			( get(parities, u ) || get(parities, v ) )
			&& !( get(parities, u ) && get(parities, v ) )
			)
		{
			std::cout << "aaa " << u+1 << " - " << v+1 << " : " << get(boost::edge_weight, aGraph)[ e ] << std::endl;
		}
	}

	std::cout << "One set of vertices consists of:" << std::endl;

	for (i = 0; i < boost::num_vertices(aGraph); ++i) {
	if (get(parities, i))
	std::cout << i+1 << " - " << boost::out_degree( i, aGraph ) << std::endl;
	}
	std::cout << std::endl;

	std::cout << "The other set of vertices consists of:" << std::endl;
	for (i = 0; i < boost::num_vertices(aGraph); ++i) {
	if (!get(parities, i))
	std::cout << i+1 << " - " << boost::out_degree( i, aGraph ) << std::endl;
	}
	std::cout << std::endl;
	D_PRINT( "Time after computeMinCut " << clock.SecondsPassed() );
}


void
computeMaxFlow( WeightedBidirectionalGraph & aGraph )
{
	/*std::vector<int> component(boost::num_vertices(aGraph));
	size_t comp = boost::connected_components(aGraph, &component[0]);

	LOG( "Number of components = " << comp );

	WeightedBidirectionalGraphTraits::vertex_descriptor src;
	WeightedBidirectionalGraphTraits::vertex_descriptor sink;
	float flow = boost::boykov_kolmogorov_max_flow( aGraph, src, sink );


	LOG( "Flow = " << flow );*/


/*	using namespace boost;

  typedef adjacency_list_traits < vecS, vecS, directedS > Traits;
  typedef adjacency_list < vecS, vecS, directedS,
    property < vertex_name_t, std::string,
    property < vertex_index_t, long,
    property < vertex_color_t, boost::default_color_type,
    property < vertex_distance_t, long,
    property < vertex_predecessor_t, Traits::edge_descriptor > > > > >,

    property < edge_capacity_t, long,
    property < edge_residual_capacity_t, long,
    property < edge_reverse_t, Traits::edge_descriptor > > > > Graph;

  Graph g;
  property_map < Graph, edge_capacity_t >::type
      capacity = get(edge_capacity, g);
  property_map < Graph, edge_residual_capacity_t >::type
      residual_capacity = get(edge_residual_capacity, g);
  property_map < Graph, edge_reverse_t >::type rev = get(edge_reverse, g);
  Traits::vertex_descriptor s, t;
  read_dimacs_max_flow(g, capacity, rev, s, t);

  std::vector<default_color_type> color(num_vertices(g));
  std::vector<long> distance(num_vertices(g));
  long flow = boykov_kolmogorov_max_flow(g ,s, t);

  std::cout << "c  The total flow:" << std::endl;
  std::cout << "s " << flow << std::endl << std::endl;

  std::cout << "c flow values:" << std::endl;
  graph_traits < Graph >::vertex_iterator u_iter, u_end;
  graph_traits < Graph >::out_edge_iterator ei, e_end;
  for (boost::tie(u_iter, u_end) = vertices(g); u_iter != u_end; ++u_iter)
    for (boost::tie(ei, e_end) = out_edges(*u_iter, g); ei != e_end; ++ei)
      if (capacity[*ei] > 0)
	std::cout << "f " << *u_iter << " " << target(*ei, g) << " "
	  << (capacity[*ei] - residual_capacity[*ei]) << std::endl;
	*/
}

/*
// A graphic of the min-cut is available at <http://www.boost.org/doc/libs/release/libs/graph/doc/stoer_wagner_imgs/stoer_wagner.cpp.gif>
int test()
{
	using namespace std;

	typedef boost::adjacency_list<
		boost::vecS,
		boost::vecS,
		boost::undirectedS,
		boost::no_property,
		boost::property<boost::edge_weight_t, float>
		> undirected_graph;

	typedef boost::graph_traits<undirected_graph>::vertex_descriptor vertex_descriptor;
	typedef boost::property_map<undirected_graph, boost::edge_weight_t>::type weight_map_type;
	typedef boost::property_traits<weight_map_type>::value_type weight_type;

	// define the 16 edges of the graph. {3, 4} means an undirected edge between vertices 3 and 4.
	edge_t edges[] = {{3, 4}, {3, 6}, {3, 5}, {0, 4}, {0, 1}, {0, 6}, {0, 7},
	{0, 5}, {0, 2}, {4, 1}, {1, 6}, {1, 5}, {6, 7}, {7, 5}, {5, 2}, {3, 4}};

	// for each of the 16 edges, define the associated edge weight. ws[i] is the weight for the edge
	// that is described by edges[i].
	weight_type ws[] = {0, 3, 1, 3, 1, 2, 6, 1, 8, 1, 1, 80, 2, 1, 1, 4};

	// construct the graph object. 8 is the number of vertices, which are numbered from 0
	// through 7, and 16 is the number of edges.
	undirected_graph g(edges, edges + 16, ws, 8, 16);

	// define a property map, `parities`, that will store a boolean value for each vertex.
	// Vertices that have the same parity after `stoer_wagner_min_cut` runs are on the same side of the min-cut.
	BOOST_AUTO(parities, boost::make_one_bit_color_map(num_vertices(g), get(boost::vertex_index, g)));

	// run the Stoer-Wagner algorithm to obtain the min-cut weight. `parities` is also filled in.
	int w = boost::stoer_wagner_min_cut(g, get(boost::edge_weight, g), boost::parity_map(parities));

	cout << "The min-cut weight of G is " << w << ".\n" << endl;
	assert(w == 7);

	cout << "One set of vertices consists of:" << endl;
	size_t i;
	for (i = 0; i < num_vertices(g); ++i) {
	if (get(parities, i))
	cout << i << endl;
	}
	cout << endl;

	cout << "The other set of vertices consists of:" << endl;
	for (i = 0; i < num_vertices(g); ++i) {
	if (!get(parities, i))
	cout << i << endl;
	}
	cout << endl;

	return EXIT_SUCCESS;
}
*/
