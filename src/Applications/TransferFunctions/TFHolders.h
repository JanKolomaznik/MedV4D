#ifndef TF_HOLDERS
#define TF_HOLDERS

namespace M4D {
namespace GUI {

namespace TF {
namespace Types {

enum Holder{
	HolderBasic
};
typedef std::vector<Holder> Holders;

static Holders getAllHolders(){

	Holders all;
	
	all.push_back(TF::Types::HolderBasic);

	return all;
}

}	//namespace Types


template<>
inline std::string convert<Types::Holder, std::string>(const Types::Holder &holder){

	switch(holder){
		case Types::HolderBasic:
		{
			return "Basic holder";
		}
	}

	tfAssert(!"Unknown holder!");
	return "Unknown holder (default)";
}

template<>
inline Types::Holder convert<std::string, Types::Holder>(const std::string &holder){

	if(holder == "Basic holder"){
		return Types::HolderBasic;
	}

	tfAssert(!"Unknown holder!");
	return Types::HolderBasic;	//default
}

}	//namespace TF

}	//namespace GUI
}	//namespace M4D

#endif	//TF_HOLDERS