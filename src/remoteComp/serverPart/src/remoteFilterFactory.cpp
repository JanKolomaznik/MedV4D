
#include "../remoteFilterFactory.h"
//#include "../remoteServerFilters/levelsetSegmentation/serverLevelsetSegmentation.h"
#include "../remoteServerFilters/levelsetSegmentation/popop.h"

using namespace M4D::RemoteComputing;
using namespace M4D::Imaging;
///////////////////////////////////////////////////////////////////////////////
M4D::Imaging::AbstractPipeFilter *
RemoteFilterFactory::DeserializeFilter(Imaging::InStream &stream, iRemoteFilterProperties **props)
{
	uint16 filterID;
	stream.Get<uint16>(filterID);
	
	switch( (FilterID) filterID)
	{
	case FID_LevelSetSegmentation:
		return CreateRemoteLevelSetSegmentationFilter(stream, props);
		break;
		
	default:
		ASSERT(false);
	}
}
///////////////////////////////////////////////////////////////////////////////
template< typename InputPixelType, typename OutputPixelType>
M4D::Imaging::AbstractPipeFilter *
CreateRemoteLevelSetSegmentationFilterStage3(iRemoteFilterProperties **props)
{
	*props = new LevelSetRemoteProperties<int16, int16>();
	return new Popop< int16, int16>( 
			(LevelSetRemoteProperties<int16, int16> *) *props);
}
///////////////////////////////////////////////////////////////////////////////
template< typename InputPixelType>
M4D::Imaging::AbstractPipeFilter *
CreateRemoteLevelSetSegmentationFilterStage2(uint16 outElemType, iRemoteFilterProperties **props)
{
	NUMERIC_TYPE_TEMPLATE_SWITCH_MACRO( outElemType,
		return CreateRemoteLevelSetSegmentationFilterStage3< InputPixelType, TTYPE>(props)		
	);
}
///////////////////////////////////////////////////////////////////////////////
M4D::Imaging::AbstractPipeFilter *
RemoteFilterFactory::CreateRemoteLevelSetSegmentationFilter(
		Imaging::InStream &stream, iRemoteFilterProperties **props)
{
	uint16 inElemType, outElemType;
	stream.Get<uint16>(inElemType);
	stream.Get<uint16>(outElemType);

	NUMERIC_TYPE_TEMPLATE_SWITCH_MACRO( 
		inElemType,
		return CreateRemoteLevelSetSegmentationFilterStage2<TTYPE>(outElemType, props)
	);
}
///////////////////////////////////////////////////////////////////////////////
