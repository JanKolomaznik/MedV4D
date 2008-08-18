// testing main for cell client

#include "Common.h"
#include "cellBE/cellClient.h"
#include "Imaging/ImageFactory.h"
#include "Imaging/PipelineContainer.h"
#include "cellBE/remoteFilters/BoneSegmentationRemote.h"

using namespace M4D::CellBE;
using namespace M4D::Imaging;

typedef Image<uint16, 3> ImageType;
int main()
{
  try
  {
	PipelineContainer pipeline;

	AbstractPipeFilter *filter = new BoneSegmentationRemote< ImageType >();

	pipeline.AddFilter( filter );
	AbstractImageConnectionInterface *inConnection = 
		dynamic_cast<AbstractImageConnectionInterface*>( &pipeline.MakeInputConnection( *filter, 0, false ) );
	AbstractImageConnectionInterface *outConnection = 
		dynamic_cast<AbstractImageConnectionInterface*>( &pipeline.MakeOutputConnection( *filter, 0, true ) );

	// prepare dataSet
	AbstractImage::AImagePtr inImage = 
		ImageFactory::CreateEmptyImage3D<uint16>(15, 15, 15);

	inConnection->PutImage( inImage );

  filter->Execute();

  ((RemoteFilterBase*)filter)->Run();
  }
  catch (std::exception& e)
  {
    std::cout << e.what() << std::endl;
    return -1;
  }

  return 0;
}
