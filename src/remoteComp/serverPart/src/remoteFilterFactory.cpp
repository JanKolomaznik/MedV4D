
#include "../remoteFilterFactory.h"
#include "../remoteServerFilters/levelsetSegmentation/serverLevelsetSegmentation.h"

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
M4D::Imaging::AbstractPipeFilter *
RemoteFilterFactory::CreateRemoteLevelSetSegmentationFilter(
		Imaging::InStream &stream)
{
	uint16 inElemType, outElemType;
	stream.Get<uint16>(inElemType);
	stream.Get<uint16>(outElemType);	

	//return new ServerLevelsetSegmentation<>();
}
///////////////////////////////////////////////////////////////////////////////
