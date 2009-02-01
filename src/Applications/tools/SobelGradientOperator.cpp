#include "Common.h"
#include "Filtering.h"
#include "Imaging/filters/SobelEdgeDetector.h"
#include <tclap/CmdLine.h>


using namespace M4D;
using namespace M4D::Imaging;

typedef Image< uint8, 2 > ImageType;
//typedef Image< SimpleVector<int16,2>, 2 > ImageGradientType;

int
main( int argc, char **argv )
{
	std::ofstream logFile( "Log.txt" );
        SET_LOUT( logFile );

        D_COMMAND( std::ofstream debugFile( "Debug.txt" ); );
        SET_DOUT( debugFile );

	TCLAP::CmdLine cmd( "Laplace operator.", ' ', "");
	/*---------------------------------------------------------------------*/

		//Define cmd arguments

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
	IMAGE_NUMERIC_TYPE_PTR_SWITCH_MACRO( image, 
		typedef Image< SimpleVector< TypeTraits< ImageTraits< IMAGE_TYPE >::ElementType >::SuperiorSignedType, 2 >, ImageTraits< IMAGE_TYPE >::Dimension > ImageGradientType;
		M4D::Imaging::SobelGradientOperator< IMAGE_TYPE, ImageGradientType > *sobel = new M4D::Imaging::SobelGradientOperator< IMAGE_TYPE, ImageGradientType >();
		filter = sobel;
	);

	/*---------------------------------------------------------------------*/
	container = PrepareSimplePipeline( *filter, M4D::Imaging::MessageReceiverInterface::Ptr( hook ), inConnection, outConnection );
	//container = PreparePipeline<ImageType>( *filter, M4D::Imaging::MessageReceiverInterface::Ptr( hook ), inConnection, outConnection );
	
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
