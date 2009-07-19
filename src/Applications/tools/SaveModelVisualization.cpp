#include "common/Common.h"
#include "Filtering.h"
#include "Imaging/filters/MedianFilter.h"
#include <tclap/CmdLine.h>
#include "Imaging/Imaging.h"

using namespace M4D;
using namespace M4D::Imaging;

typedef M4D::Imaging::Image< int16, 3 > ImageType;


struct InModelVisualizationAccessor
{
	InModelVisualizationAccessor( unsigned val ): _val( val ) {}

	int16
	operator()( const GridPointRecord & rec ) 
	{
		return (int16)( 4096 * rec.inHistogram[ _val ] );
	}
	unsigned _val;
};

struct OutModelVisualizationAccessor
{
	int16
	operator()( const GridPointRecord & rec ) 
	{
		return (int16)( 4096 * rec.inProbabilityPos );
	}
};


int
main( int argc, char **argv )
{
	std::ofstream logFile( "Log.txt" );
        SET_LOUT( logFile );

        D_COMMAND( std::ofstream debugFile( "Debug.txt" ); );
        SET_DOUT( debugFile );
	
	TCLAP::CmdLine cmd( "Model visualizer.", ' ', "");
	/*---------------------------------------------------------------------*/

	TCLAP::ValueArg<unsigned> inHistogramValArg( "i", "inHistogram", "inHistogram", false, 1000, "Unsigned integer" );
	cmd.add( inHistogramValArg );

	TCLAP::ValueArg<unsigned> outHistogramValArg( "o", "outHistogram", "outHistogram", false, 1000, "Unsigned integer" );
	cmd.add( outHistogramValArg );

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
	M4D::Imaging::CanonicalProbModel::Ptr probModel = CanonicalProbModel::LoadFromFile( inFilename );
	std::cout << "Done\n";


	std::cout << "Creating image..."; std::cout.flush();
	ImageType::Ptr tmp;
	if ( inHistogramValArg.isSet() ) {
	 	tmp = MakeImageFromProbabilityGrid<InModelVisualizationAccessor>( probModel->GetGrid(), InModelVisualizationAccessor( inHistogramValArg.getValue() ) );
	} else {
		std::cerr << "Unfinished options !!!\n";
		return 1;
	}
	//TODO
	std::cout << "Done\n";

	std::cout << "Saving file '" << outFilename << "' ..."; std::cout.flush();
	ImageFactory::DumpImage( outFilename, *tmp );
	std::cout << "Done\n";

	return 0;
}
