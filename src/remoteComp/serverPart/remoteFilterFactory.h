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
		
	// class ID serialization & deserialization
	static void SerializeFilterClassID(
			M4D::IO::OutStream &stream, const AbstractFilter &filter);	
	static AbstractFilter::Ptr 
		DeserializeFilterClassID(M4D::IO::InStream &stream);
	
	// Actual properties content serialization & deserialization	
	static void SerializeFilterProperties(
			M4D::IO::OutStream &stream, const AbstractFilter &filter);		
	static void	DeSerializeFilterProperties(
			M4D::IO::InStream &stream, AbstractFilter &filter);
	
private:
	static M4D::Imaging::AbstractPipeFilter *
		CreateRemoteLevelSetSegmentationFilter(M4D::IO::InStream &stream, iRemoteFilterProperties **props);
};

}
}
#endif /*REMOTEFILTERFACTORY_*/
