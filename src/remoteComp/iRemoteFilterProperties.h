#ifndef IREMOTEFILTERPROPERTIES_H_
#define IREMOTEFILTERPROPERTIES_H_

#include "Imaging/IO/IOStreams.h"

namespace M4D
{
namespace RemoteComputing
{

class iRemoteFilterProperties
{
public:
	virtual void SerializeClassInfo(Imaging::OutStream &stream) = 0;
	virtual void SerializeProperties(Imaging::OutStream &stream) = 0;
	virtual void DeserializeProperties(Imaging::InStream &stream) = 0;
};

}
}
#endif /*IREMOTEFILTERPROPERTIES_H_*/
