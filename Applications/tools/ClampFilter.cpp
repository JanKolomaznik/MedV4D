#include "MedV4D/Common/Common.h"
#include "Filtering.h"
#include "Imaging/filters/ClampFilter.h"
#undef min
#undef max
#include <tclap/CmdLine.h>

using namespace M4D;
using namespace M4D::Imaging;

typedef Image< int16, 2 > ImageType;

int
main( int argc, char **argv )
{
	std::ofstream logFile( "Log.txt" );
        SET_LOUT( logFile );

        D_COMMAND( std::ofstream debugFile( "Debug.txt" ); );
        SET_DOUT( debugFile );

	TCLAP::CmdLine cmd( "Clamp filter.", ' ', "");
	/*---------------------------------------------------------------------*/

		//Define cmd arguments
		TCLAP::ValueArg<double> bottomArg( "b", "bottom", "Bottom clamp value", false, 0, "" );
		cmd.add( bottomArg );

		TCLAP::ValueArg<double>topArg( "t", "top", "Top clamp value", false, 2047, "" );
		cmd.add( topArg );

	/*---------------------------------------------------------------------*/
	TCLAP::UnlabeledValueArg<std::string> inFilenameArg( "input", "Input image filename", true, "", "filename1" );
	cmd.add( inFilenameArg );

	TCLAP::UnlabeledValueArg<std::string> outFilenameArg( "output", "Output image filename", true, "", "filename2" );
	cmd.add( outFilenameArg );

	cmd.parse( argc, argv );

	/***************************************************/

	std::string inFilename = inFilenameArg.getValue();
	std::string outFilename = outFilenameArg.getValue();

	std::cout << "Loading file '" << inFilename << "' ..."; std::cout.flush();
	M4D::Imaging::AImage::Ptr image = 
			M4D::Imaging::ImageFactory::LoadDumpedImage( inFilename );
	std::cout << "Done\n";


	std::cout << "Initializing..."; std::cout.flush();
	M4D::Imaging::PipelineContainer *container = NULL;
	FinishHook  *hook = new FinishHook;
	M4D::Imaging::ConnectionInterfaceTyped< M4D::Imaging::AImage > *inConnection = NULL;
	M4D::Imaging::ConnectionInterfaceTyped< M4D::Imaging::AImage > *outConnection = NULL;
	M4D::Imaging::APipeFilter *filter = NULL;
	/*---------------------------------------------------------------------*/
	double bottom = bottomArg.getValue();
	double top = topArg.getValue();
	IMAGE_NUMERIC_TYPE_PTR_SWITCH_MACRO( image, 
		M4D::Imaging::ClampFilter<IMAGE_TYPE> *tfilter = new M4D::Imaging::ClampFilter<IMAGE_TYPE>();
		tfilter->SetBottom( static_cast<TTYPE>( bottom ) );
		tfilter->SetTop( static_cast<TTYPE>( top ) );
		filter = tfilter;
	);

	/*---------------------------------------------------------------------*/
	container = PrepareSimplePipeline( *filter, M4D::Imaging::MessageReceiverInterface::Ptr( hook ), inConnection, outConnection );
	//container = PreparePipeline<ImageType>( *filter, M4D::Imaging::MessageReceiverInterface::Ptr( hook ), inConnection, outConnection );
	
	inConnection->PutDataset( image );

	std::cout << "Done\n";

	std::cout << "Computing..."; std::cout.flush();
	container->ExecuteFirstFilter();

	while( !(hook->Finished()) ){ /*empty*/ }
	if( hook->OK() ) {
		std::cout << "Done in "<< filter->GetLastComputationTime() << " seconds.\n";

		std::cout << "Saving file '" << outFilename << "' ..."; std::cout.flush();
		M4D::Imaging::ImageFactory::DumpImage( outFilename, outConnection->GetDatasetReadOnlyTyped() );
		std::cout << "Done\n";
	} else {
		std::cout << "FAILED\n";
	}
	delete container;

	return 0;
}
