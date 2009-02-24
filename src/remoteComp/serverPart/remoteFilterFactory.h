#ifndef REMOTEFILTERFACTORY_
#define REMOTEFILTERFACTORY_

#include "Imaging/AbstractFilter.h"
#include "Imaging/IO/IOStreams.h"

namespace M4D
{
namespace RemoteComputing
{

class RemoteFilterFactory
{
public:
	static M4D::Imaging::AbstractPipeFilter *
		DeserializeFilter(Imaging::InStream &stream);
	
private:
	static M4D::Imaging::AbstractPipeFilter *
		CreateRemoteLevelSetSegmentationFilter(Imaging::InStream &stream);
};

}
}
#endif /*REMOTEFILTERFACTORY_*/
