#ifndef LEVELSETREMOTEPROPERTIES_H_
#define LEVELSETREMOTEPROPERTIES_H_

#include "remoteComp/iRemoteFilterProperties.h"
#include "remoteComp/filterIDEnums.h"

namespace M4D
{
namespace RemoteComputing
{

template< typename InputElementType, typename OutputElementType >
class LevelSetRemoteProperties
	: public iRemoteFilterProperties
{
public:
	LevelSetRemoteProperties()
		: seedX( 128 )
		, seedY( 128 )
		, seedZ( 1 )
		, lowerThreshold( -500 ) 
		, upperThreshold( 500 ) 
		, maxIterations( 800 ) 
		, initialDistance( 5.0f ) 
		, curvatureScaling( 0.01f ) 
		, propagationScaling( 1.0f ) 
		, advectionScaling( 10.0f ) 
	{}
	
	virtual ~LevelSetRemoteProperties() {}

	uint32 seedX;
	uint32 seedY;
	uint32 seedZ;
	InputElementType lowerThreshold;
	InputElementType upperThreshold;
	uint32 maxIterations;
	float32 initialDistance;
	float32 curvatureScaling;
	float32 propagationScaling;
	float32 advectionScaling;

	void
	CheckProperties() {
		maxIterations >= 50;
		initialDistance >= 3;
	}
	
	void SerializeClassInfo(M4D::IO::OutStream &stream)
	{
		stream.Put<uint16>(FID_LevelSetSegmentation);
		stream.Put<uint16>(GetNumericTypeID< InputElementType >());
		stream.Put<uint16>(GetNumericTypeID< OutputElementType >());
	}
	void SerializeProperties(M4D::IO::OutStream &stream)
	{
		stream.Put<uint32>(seedX);
		stream.Put<uint32>(seedY);
		stream.Put<uint32>(seedZ);
		stream.Put<InputElementType>(lowerThreshold);
		stream.Put<InputElementType>(upperThreshold);
		stream.Put<uint32>(maxIterations);
		stream.Put<float32>(initialDistance);
		stream.Put<float32>(curvatureScaling);
		stream.Put<float32>(propagationScaling);
		stream.Put<float32>(advectionScaling);
	}
	void DeserializeProperties(M4D::IO::InStream &stream)
	{
		stream.Get<uint32>(seedX);
		stream.Get<uint32>(seedY);
		stream.Get<uint32>(seedZ);
		stream.Get<InputElementType>(lowerThreshold);
		stream.Get<InputElementType>(upperThreshold);
		stream.Get<uint32>(maxIterations);
		stream.Get<float32>(initialDistance);
		stream.Get<float32>(curvatureScaling);
		stream.Get<float32>(propagationScaling);
		stream.Get<float32>(advectionScaling);
	}
};

}
}
#endif /*LEVELSETREMOTEFILTER_H_*/
