#ifndef REMOTEFILTERFACTORY_
#define REMOTEFILTERFACTORY_

#include "Imaging/AbstractFilter.h"
#include "common/IOStreams.h"
#include "remoteComp/iRemoteFilterProperties.h"

namespace M4D
{
namespace RemoteComputing
{

class RemoteFilterFactory
{
public:
	static M4D::Imaging::AbstractPipeFilter *
		DeserializeFilter(M4D::IO::InStream &stream, iRemoteFilterProperties **props);
	
private:
	static M4D::Imaging::AbstractPipeFilter *
		CreateRemoteLevelSetSegmentationFilter(M4D::IO::InStream &stream, iRemoteFilterProperties **props);
};

}
}
#endif /*REMOTEFILTERFACTORY_*/
