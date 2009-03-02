#ifndef REMOTEFILTERFACTORY_
#define REMOTEFILTERFACTORY_

#include "Imaging/AbstractFilter.h"
#include "Imaging/IO/IOStreams.h"
#include "remoteComp/iRemoteFilterProperties.h"

namespace M4D
{
namespace RemoteComputing
{

class RemoteFilterFactory
{
public:
	static M4D::Imaging::AbstractPipeFilter *
		DeserializeFilter(Imaging::InStream &stream, iRemoteFilterProperties **props);
	
private:
	static M4D::Imaging::AbstractPipeFilter *
		CreateRemoteLevelSetSegmentationFilter(Imaging::InStream &stream, iRemoteFilterProperties **props);
};

}
}
#endif /*REMOTEFILTERFACTORY_*/
