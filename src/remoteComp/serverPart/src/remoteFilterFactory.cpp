
#include "common/Common.h"
#include "../remoteFilterFactory.h"
#include "remoteComp/remoteServerFilters/levelsetSegmentation/medevedWrapperFilter.h"
//#include "remoteComp/remoteFilterProperties/thresholdingRemoteProperties.h"
//#include "remoteComp/remoteServerFilters/levelsetSegmentation2/medevedWrapperFilter.h"

using namespace M4D::RemoteComputing;
using namespace M4D::Imaging;
using namespace M4D::IO;

typedef M4D::Imaging::Image<int16, 3> TImage;

///////////////////////////////////////////////////////////////////////////////
template< typename InputPixelType, typename OutputPixelType>
M4D::Imaging::AbstractPipeFilter *
CreateFilterStage2(uint16 filterID, iRemoteFilterProperties **props)
{
	switch( (FilterID) filterID)
	{
	case FID_LevelSetSegmentation:
		*props = new LevelSetRemoteProperties<int16, int16>();
		return new ThreshLSSegMedvedWrapper< int16, int16>( 
				dynamic_cast<LevelSetRemoteProperties<int16, int16> *>(*props) );
		break;
		
//	case FID_Thresholding:
//			*props = new ThresholdingRemoteProps<int16, int16>();
//			return new M4D::Imaging::ThresholdingFilter<TImage>( 
//					dynamic_cast<M4D::Imaging::ThresholdingFilter<TImage>::Properties *>(*props) );
//			break;
		
	default:
		ASSERT(false);
	}
}
///////////////////////////////////////////////////////////////////////////////
template< typename InputPixelType>
M4D::Imaging::AbstractPipeFilter *
CreateFilterStage1(uint16 outElemType, uint16 filterID, iRemoteFilterProperties **props)
{
	NUMERIC_TYPE_TEMPLATE_SWITCH_MACRO( outElemType,
		return CreateFilterStage2< InputPixelType, TTYPE>(filterID, props)		
	);
}
///////////////////////////////////////////////////////////////////////////////
M4D::Imaging::AbstractPipeFilter *
RemoteFilterFactory::DeserializeFilter(InStream &stream, iRemoteFilterProperties **props)
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
}
///////////////////////////////////////////////////////////////////////////////
