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
		: seedX( 0 )
		, seedY( 0 )
		, seedZ( 0 )
		, lowerThreshold( 0 ) 
		, upperThreshold( 0 ) 
		, maxIterations( 50 ) 
		, initialDistance( 3 ) 
		, curvatureScaling( 1 ) 
		, propagationScaling( 0 ) 
		, advectionScaling( 0 ) 
	{}

	InputElementType seedX;
	InputElementType seedY;
	InputElementType seedZ;
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
	
	void SerializeClassInfo(Imaging::OutStream &stream)
	{
		stream.Put<uint16>(FID_LevelSetSegmentation);
		stream.Put<uint16>(GetNumericTypeID< InputElementType >());
		stream.Put<uint16>(GetNumericTypeID< OutputElementType >());
	}
	void SerializeProperties(Imaging::OutStream &stream)
	{
		stream.Put<InputElementType>(seedX);
		stream.Put<InputElementType>(seedY);
		stream.Put<InputElementType>(seedZ);
		stream.Put<InputElementType>(lowerThreshold);
		stream.Put<InputElementType>(upperThreshold);
		stream.Put<uint32>(maxIterations);
		stream.Put<float32>(initialDistance);
		stream.Put<float32>(curvatureScaling);
		stream.Put<float32>(propagationScaling);
		stream.Put<float32>(advectionScaling);
	}
	void DeserializeProperties(Imaging::InStream &stream)
	{
		stream.Get<InputElementType>(seedX);
		stream.Get<InputElementType>(seedY);
		stream.Get<InputElementType>(seedZ);
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
