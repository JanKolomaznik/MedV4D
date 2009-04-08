#ifndef THRESHOLDINGREMOTEFILTERPROPERTIES_H_
#define THRESHOLDINGREMOTEFILTERPROPERTIES_H_

#include "remoteComp/iRemoteFilterProperties.h"
#include "remoteComp/filterIDEnums.h"
#include "Imaging/filters/ThresholdingFilter.h"

namespace M4D
{
namespace RemoteComputing
{

template< typename InputElementType, typename OutputElementType >
class ThresholdingRemoteProps
	: public iRemoteFilterProperties
	, public M4D::Imaging::ThresholdingFilter<M4D::Imaging::Image<InputElementType, 3> >::Properties
{
public:
	ThresholdingRemoteProps()
	{
		this->bottom = -500; 
		this->top = 500;
	}

//
//	void
//	CheckProperties() {
//		if(this->bottom < this->top)
//			this->top = this->bottom + 10;
//	}
	
	void SerializeClassInfo(M4D::IO::OutStream &stream)
	{
		stream.Put<uint16>(FID_Thresholding);
		stream.Put<uint16>(GetNumericTypeID< InputElementType >());
		stream.Put<uint16>(GetNumericTypeID< OutputElementType >());
	}
	void SerializeProperties(M4D::IO::OutStream &stream)
	{
		stream.Put<InputElementType>(this->bottom);
		stream.Put<InputElementType>(this->top);
	}
	void DeserializeProperties(M4D::IO::InStream &stream)
	{
		stream.Get<InputElementType>(this->bottom);
		stream.Get<InputElementType>(this->top);
	}
};

}
}

#endif /*THRESHOLDINGREMOTEFILTERPROPERTIES_H_*/
