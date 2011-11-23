#ifndef TF_DIMENSIONS
#define TF_DIMENSIONS

namespace M4D {
namespace GUI {

namespace TF {
namespace Types {

enum Dimension{
	Dimension1 = 1
};
typedef std::vector<Dimension> Dimensions;

static Dimensions getSupportedDimensions(){

	Dimensions all;
	
	all.push_back(Dimension1);

	return all;
}

}	//namespace Types


template<>
inline std::string convert<Types::Dimension, std::string>(const Types::Dimension &dimension){

	switch(dimension){
		case Types::Dimension1:
		{
			return "1D";
		}
	}

	tfAssert(!"Unknown dimension!");
	return "Unknown dimension (default)";
}

template<>
inline Types::Dimension convert<std::string, Types::Dimension>(const std::string &dimension){

	if(dimension == "1D"){
		return Types::Dimension1;
	}

	tfAssert(!"Unknown dimension!");
	return Types::Dimension1;	//default
}

}	//namespace TF

}	//namespace GUI
}	//namespace M4D

#endif	//TF_DIMENSIONS