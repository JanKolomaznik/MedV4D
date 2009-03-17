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
		ImageFactory::CreateEmptyImage3DTyped<ElementType>(8, 8, 8);
	for( unsigned i = 0; i < 8; ++i ) {
		for( unsigned j = 0; j < 8; ++j ) {
			for( unsigned k = 0; k < 8; ++k ) {
				inImage->GetElement( i, j, k ) = j;//(i) | (j >> 3) | (k >> 6);
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

	BoneSegmentationRemote< ImageType > *filter = new BoneSegmentationRemote< ImageType >();

	pipeline.AddFilter( filter );
	AbstractImageConnectionInterface *inConnection = 
		dynamic_cast<AbstractImageConnectionInterface*>( &pipeline.MakeInputConnection( *filter, 0, false ) );
	AbstractImageConnectionInterface *outConnection = 
		dynamic_cast<AbstractImageConnectionInterface*>( &pipeline.MakeOutputConnection( *filter, 0, true ) );

	// prepare dataSet
	ImageType::Ptr inImage = GenerateDataset();

	inConnection->PutImage( inImage );

  filter->GetMedianOptions()->radius = 0;
  filter->GetThreshholdingOptions()->bottom = 3;
  filter->GetThreshholdingOptions()->top = 5;

  filter->Execute();

  D_PRINT( "Result:" << std::endl << std::endl);
  D_COMMAND( outConnection->GetDataset().Dump() );
  
  }
  catch (std::exception& e)
  {
    std::cout << e.what() << std::endl;
    return -1;
  }
  catch( ...)
  {
    LOG( "FATAL ERROR!");
  }

  return 0;
}
