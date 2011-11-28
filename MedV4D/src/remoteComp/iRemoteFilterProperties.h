#ifndef IREMOTEFILTERPROPERTIES_H_
#define IREMOTEFILTERPROPERTIES_H_

#include "MedV4D/Common/IOStreams.h"

namespace M4D
{
namespace RemoteComputing
{

class iRemoteFilterProperties
{
public:
	virtual ~iRemoteFilterProperties() {}
	
	virtual void SerializeClassInfo(M4D::IO::OutStream &stream) const = 0;
	virtual void SerializeProperties(M4D::IO::OutStream &stream) const = 0;
	virtual void DeserializeProperties(M4D::IO::InStream &stream) = 0;
};

}
}
#endif /*IREMOTEFILTERPROPERTIES_H_*/
