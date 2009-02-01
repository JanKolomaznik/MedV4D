#include "Common.h"
#include "Filtering.h"
#include "Imaging/filters/GaussianFilter.h"
#include <tclap/CmdLine.h>


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
	
	TCLAP::CmdLine cmd( "Gaussian filter - smooth.", ' ', "");
	/*---------------------------------------------------------------------*/

	TCLAP::ValueArg<unsigned> radiusArg( "r", "radius", "Filter mask radius.", false, 5, "Unsigned integer" );
	cmd.add( radiusArg );

	/*---------------------------------------------------------------------*/
	TCLAP::UnlabeledValueArg<std::string> inFilenameArg( "input", "Input image filename", true, "", "filename1" );
	cmd.add( inFilenameArg );

	TCLAP::UnlabeledValueArg<std::string> outFilenameArg( "output", "Output image filename", true, "", "filename2" );
	cmd.add( outFilenameArg );

	cmd.parse( argc, argv );

	/***************************************************/

	std::string inFilename = inFilenameArg.getValue();
	std::string outFilename = outFilenameArg.getValue();

	std::cout << "Loading file..."; std::cout.flush();
	M4D::Imaging::AbstractImage::Ptr image = 
			M4D::Imaging::ImageFactory::LoadDumpedImage( inFilename );
	std::cout << "Done\n";


	std::cout << "Initializing..."; std::cout.flush();
	M4D::Imaging::PipelineContainer *container = NULL;
	FinishHook  *hook = new FinishHook;
	M4D::Imaging::AbstractImageConnectionInterface *inConnection = NULL;
	M4D::Imaging::AbstractImageConnectionInterface *outConnection = NULL;
	M4D::Imaging::AbstractPipeFilter *filter = NULL;
	/*---------------------------------------------------------------------*/
	unsigned radius = radiusArg.getValue();

	IMAGE_NUMERIC_TYPE_PTR_SWITCH_MACRO( image, 
		M4D::Imaging::GaussianFilter2D< IMAGE_TYPE > *gaussFilter = new M4D::Imaging::GaussianFilter2D< IMAGE_TYPE >();
		gaussFilter->SetRadius( radius );
		filter = gaussFilter;
	);

	/*---------------------------------------------------------------------*/
	container = PrepareSimplePipeline( *filter, M4D::Imaging::MessageReceiverInterface::Ptr( hook ), inConnection, outConnection );
	//container = PreparePipeline<ImageType>( *filter, M4D::Imaging::MessageReceiverInterface::Ptr( hook ), inConnection, outConnection );
	//
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
