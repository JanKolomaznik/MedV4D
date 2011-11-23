#include "MedV4D/Common/Common.h"
#include "Filtering.h"
#include "Imaging/filters/MedianFilter.h"
#undef min
#undef max
#include <tclap/CmdLine.h>
#include "Imaging/Imaging.h"

using namespace M4D;
using namespace M4D::Imaging;

typedef M4D::Imaging::Image< int16, 3 > ImageType;


struct ModelVisualizationAccessor
{
	ModelVisualizationAccessor( bool inSet, unsigned inVal,  bool outSet, unsigned outVal )
		: _inSet( inSet ), _inVal( inVal ), _outSet( outSet ), _outVal( outVal ) {}

	int16
	operator()( const GridPointRecord & rec ) 
	{
		int16 result = 0;
		if( _inSet ) {
			result += (int16)( 4096 * rec.inHistogram[ _inVal ] );
		}
		if( _outSet ) {
			result += (int16)( 4096 * rec.outHistogram[ _outVal ] );
		}
		return result;
	}
	bool		_inSet;
	unsigned 	_inVal;
	bool		_outSet;
	unsigned	_outVal;
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
	tmp = MakeImageFromProbabilityGrid<ModelVisualizationAccessor>( 
			probModel->GetGrid(), 
			ModelVisualizationAccessor( inHistogramValArg.isSet(), inHistogramValArg.getValue(), outHistogramValArg.isSet(), outHistogramValArg.getValue() ) 
			);
	//TODO
	std::cout << "Done\n";

	std::cout << "Saving file '" << outFilename << "' ..."; std::cout.flush();
	ImageFactory::DumpImage( outFilename, *tmp );
	std::cout << "Done\n";

	return 0;
}
