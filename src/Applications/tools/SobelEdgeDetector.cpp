#include "Common.h"
#include "Filtering.h"
#include "Imaging/filters/SobelEdgeDetector.h"


using namespace M4D;
using namespace M4D::Imaging;

typedef Image< uint8, 2 > ImageType;

int
main( int argc, char **argv )
{
	std::ofstream logFile( "Log.txt" );
        SET_LOUT( logFile );

        D_COMMAND( std::ofstream debugFile( "Debug.txt" ); );
        SET_DOUT( debugFile );

	if( argc < 3 || argc > 3 ) {
                std::cerr << "Wrong argument count - must be in form: 'program inputfile outputfile'\n";
                return 1;
        }

	std::string inFilename = argv[1];
	std::string outFilename = argv[2];


	std::cout << "Loading file..."; std::cout.flush();
	M4D::Imaging::AbstractImage::AImagePtr image = 
			M4D::Imaging::ImageFactory::LoadDumpedImage( inFilename );
	std::cout << "Done\n";


	std::cout << "Initializing..."; std::cout.flush();
	M4D::Imaging::PipelineContainer *container = NULL;
	FinishHook  *hook = new FinishHook;
	M4D::Imaging::AbstractImageConnectionInterface *inConnection = NULL;
	M4D::Imaging::AbstractImageConnectionInterface *outConnection = NULL;
	/*---------------------------------------------------------------------*/
	M4D::Imaging::SobelEdgeDetector< ImageType > *filter = new M4D::Imaging::SobelEdgeDetector< ImageType >();

	/*---------------------------------------------------------------------*/
	container = PreparePipeline<ImageType>( *filter, M4D::Imaging::MessageReceiverInterface::Ptr( hook ), inConnection, outConnection );
	inConnection->PutImage( image );

	std::cout << "Done\n";

	std::cout << "Computing..."; std::cout.flush();
	container->ExecuteFirstFilter();

	while( !(hook->Finished()) ){ /*empty*/ }
	if( hook->OK() ) {
		std::cout << "Done\n";

		std::cout << "Saving file..."; std::cout.flush();
		M4D::Imaging::ImageFactory::DumpImage( outFilename, outConnection->GetAbstractImageReadOnly() );
		std::cout << "Done\n";
	} else {
		std::cout << "FAILED\n";
	}
	delete container;

	return 0;
}
