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
	
	class ENotRemoteFilter;
		
	// class ID serialization & deserialization
	static void SerializeFilterClassID(
			M4D::IO::OutStream &stream, const iRemoteFilterProperties &props);	
	static M4D::Imaging::AbstractPipeFilter *
		DeserializeFilterClassID(M4D::IO::InStream &stream, iRemoteFilterProperties **props);
	
	// Actual properties content serialization & deserialization	
	static void SerializeFilterProperties(
			M4D::IO::OutStream &stream, const iRemoteFilterProperties &props);	
	static void	DeSerializeFilterProperties(
			M4D::IO::InStream &stream, iRemoteFilterProperties &props);
};

class RemoteFilterFactory::ENotRemoteFilter 
	: public M4D::ErrorHandling::ExceptionBase
{
public:
	ENotRemoteFilter() {}

	//TODO
};

}
}
#endif /*REMOTEFILTERFACTORY_*/
