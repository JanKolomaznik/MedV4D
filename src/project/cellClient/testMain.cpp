// testing main for cell client

#include "Common.h"
#include "cellBE/cellClient.h"
#include "Imaging/ImageFactory.h"
#include "Imaging/PipelineContainer.h"
#include "cellBE/remoteFilters/BoneSegmentationRemote.h"

using namespace M4D::CellBE;
using namespace M4D::Imaging;

typedef uint16 ElementType;
typedef Image<uint16, 3> ImageType;


ImageType::Ptr
GenerateDataset()
{
	ImageType::Ptr inImage = 
		ImageFactory::CreateEmptyImage3D<ElementType>(16, 16, 16);
	for( unsigned i = 0; i < 16; ++i ) {
		for( unsigned j = 0; j < 16; ++j ) {
			for( unsigned k = 0; k < 16; ++k ) {
				inImage->GetElement( i, j, k ) = (i << 2) | (j << 1) | k;
			}
		}
	}
	return inImage;
}


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
	ImageType::Ptr inImage = GenerateDataset();

	inConnection->PutImage( inImage );

  filter->Execute();

  ((RemoteFilterBase*)filter)->Run();

  while( 1)
  {
  }

  }
  catch (std::exception& e)
  {
    std::cout << e.what() << std::endl;
    return -1;
  }

  return 0;
}
