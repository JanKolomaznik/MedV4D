
#include "../remoteFilterFactory.h"
//#include "../remoteServerFilters/levelsetSegmentation/serverLevelsetSegmentation.h"
#include "../remoteServerFilters/levelsetSegmentation/popop.h"

using namespace M4D::RemoteComputing;
using namespace M4D::Imaging;
///////////////////////////////////////////////////////////////////////////////
M4D::Imaging::AbstractPipeFilter *
RemoteFilterFactory::DeserializeFilter(Imaging::InStream &stream)
{
	uint16 filterID;
	stream.Get<uint16>(filterID);
	
	switch( (FilterID) filterID)
	{
	case FID_LevelSetSegmentation:
		return CreateRemoteLevelSetSegmentationFilter(stream);
		break;
		
	default:
		ASSERT(false);
	}
}
///////////////////////////////////////////////////////////////////////////////
template< typename InputPixelType>
M4D::Imaging::AbstractPipeFilter *
CreateRemoteLevelSetSegmentationFilterStage2(uint16 outElemType)
{
//	NUMERIC_TYPE_TEMPLATE_SWITCH_MACRO( outElemType,
//		return new ServerLevelsetSegmentation< InputPixelType, TTYPE>()	
//	);
	return NULL;
}
///////////////////////////////////////////////////////////////////////////////
M4D::Imaging::AbstractPipeFilter *
RemoteFilterFactory::CreateRemoteLevelSetSegmentationFilter(
		Imaging::InStream &stream)
{
	uint16 inElemType, outElemType;
	stream.Get<uint16>(inElemType);
	stream.Get<uint16>(outElemType);

//	NUMERIC_TYPE_TEMPLATE_SWITCH_MACRO( 
//		inElemType,
//		return CreateRemoteLevelSetSegmentationFilterStage2<TTYPE>(outElemType)
//	);
	return new Popop< uint16, uint16>();
}
///////////////////////////////////////////////////////////////////////////////
