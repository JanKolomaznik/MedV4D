#include "Common.h"
#include "Filtering.h"


using namespace M4D;
using namespace M4D::Imaging;

typedef Image< int16, 3 > ImageType;

int
main( int argc, char **argv )
{

	std::string inFilename = argv[1];
	std::string outFilename = argv[2];


	std::cout << "Loading file...";
	M4D::Imaging::AbstractImage::AImagePtr image = 
			M4D::Imaging::ImageFactory::LoadDumpedImage( inFilename );
	std::cout << "Done\n";


	M4D::Imaging::PipelineContainer *container = NULL;
	M4D::Imaging::MessageReceiverInterface::Ptr hook;
	M4D::Imaging::AbstractImageConnectionInterface *inConnection = NULL;
	M4D::Imaging::AbstractImageConnectionInterface *outConnection = NULL;

	container = PreparePipeline<ImageType>( &filter, hook, inConnection, *outConnection );

	std::cout << "Loading file...";
	M4D::Imaging::ImageFactory::DumpImage( outFileName, *image );
	std::cout << "Done\n";

	return 0;
}
