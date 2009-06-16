
#include "common/Common.h"
#include "../remoteFilterFactory.h"
#include "remoteComp/remoteServerFilters/levelsetSegmentation/medevedWrapperFilter.h"
//#include "remoteComp/remoteFilterProperties/thresholdingRemoteProperties.h"
//#include "remoteComp/remoteServerFilters/levelsetSegmentation2/medevedWrapperFilter.h"

using namespace M4D::RemoteComputing;
using namespace M4D::Imaging;
using namespace M4D::IO;

//typedef M4D::Imaging::Image<int16, 3> TImage;

///////////////////////////////////////////////////////////////////////////////
template< typename InputPixelType, typename OutputPixelType>
M4D::Imaging::AbstractPipeFilter *
CreateFilterStage2(uint16 filterID, iRemoteFilterProperties **props)
{
	switch( (FilterID) filterID)
	{
	case FID_LevelSetSegmentation:
		typedef ThreshLSSegMedvedWrapper<float32, float32> Filter;
		*props = new Filter::Properties();
		return new Filter(dynamic_cast<Filter::Properties *>(*props));
		break;
		
//	case FID_Thresholding:
//			*props = new ThresholdingRemoteProps<int16, int16>();
//			return new M4D::Imaging::ThresholdingFilter<TImage>( 
//					dynamic_cast<M4D::Imaging::ThresholdingFilter<TImage>::Properties *>(*props) );
//			break;
		
	default:
		ASSERT(false);
	}
	return NULL;	// just to remove warns
}
///////////////////////////////////////////////////////////////////////////////
template< typename InputPixelType>
M4D::Imaging::AbstractPipeFilter *
CreateFilterStage1(
		uint16 outElemType, uint16 filterID, iRemoteFilterProperties **props)
{
	NUMERIC_TYPE_TEMPLATE_SWITCH_MACRO( outElemType,
		return CreateFilterStage2< InputPixelType, TTYPE>(filterID, props)		
	);
	return NULL;	// just to remove warns
}
///////////////////////////////////////////////////////////////////////////////
AbstractPipeFilter* 
RemoteFilterFactory::DeserializeFilterClassID(
		M4D::IO::InStream &stream, iRemoteFilterProperties **props)
{
	uint16 filterID;
	uint16 inElemType, outElemType;
	
	stream.Get<uint16>(filterID);
	stream.Get<uint16>(inElemType);
	stream.Get<uint16>(outElemType);
	
	NUMERIC_TYPE_TEMPLATE_SWITCH_MACRO( 
			inElemType,
			return CreateFilterStage1<TTYPE>(outElemType, filterID, props)
		);
	return NULL;	// just to remove warns
}

///////////////////////////////////////////////////////////////////////////////
void
RemoteFilterFactory::SerializeFilterClassID(
		M4D::IO::OutStream &stream, const iRemoteFilterProperties &props)
{
		props.SerializeClassInfo(stream);
}
///////////////////////////////////////////////////////////////////////////////
void
RemoteFilterFactory::SerializeFilterProperties(
		M4D::IO::OutStream &stream, const iRemoteFilterProperties &props)
{
			props.SerializeProperties(stream);
}
///////////////////////////////////////////////////////////////////////////////
void
RemoteFilterFactory::DeSerializeFilterProperties(
		M4D::IO::InStream &stream, iRemoteFilterProperties &props)
{		
		props.DeserializeProperties(stream);		
}

///////////////////////////////////////////////////////////////////////////////
