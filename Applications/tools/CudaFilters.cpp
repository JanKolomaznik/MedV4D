#include "MedV4D/Common/Common.h"
#include "MedV4D/Imaging/Imaging.h"
#undef min
#undef max
#include <tclap/CmdLine.h>
#include "MedV4D/Imaging/cuda/ConnectedComponentLabeling.h"
#include "MedV4D/Imaging/cuda/EdgeDetection.h"
#include "MedV4D/Imaging/cuda/LocalMinimaDetection.h"
#include "MedV4D/Imaging/cuda/WatershedTransformation.h"
#include "MedV4D/Imaging/cuda/GraphOperations.h"
#include "MedV4D/Imaging/cuda/SimpleFilters.h"

#include <boost/algorithm/string.hpp>



using namespace M4D;
using namespace M4D::Imaging;


int
main( int argc, char **argv )
{
//	std::ofstream logFile( "Log.txt" );
//        SET_LOUT( logFile );

//        D_COMMAND( std::ofstream debugFile( "Debug.txt" ); );
//        SET_DOUT( debugFile );

	TCLAP::CmdLine cmd( "Sobel operator.", ' ', "");
	/*---------------------------------------------------------------------*/

		//Define cmd arguments
	TCLAP::ValueArg<std::string> operatorTypeArg( "o", "operator", "Applied operator", true, "", "Name of applied operator (SOBEL, MIN, BORDERS, CCL, WSHED/WATERSHED, WSHED2/WATERSHED2" );
	cmd.add( operatorTypeArg );

	TCLAP::ValueArg<double> thresholdArg( "t", "threshold", "Edge threshold value", false, 0, "Numeric type of image element" );
	cmd.add( thresholdArg );
	
	TCLAP::ValueArg<double> belowThresholdArg( "b", "below", "Below threshold value", false, 0, "Numeric type of image element" );
	cmd.add( belowThresholdArg );

	/*---------------------------------------------------------------------*/
	TCLAP::UnlabeledValueArg<std::string> inFilenameArg( "input", "Input image filename", true, "", "filename1" );
	cmd.add( inFilenameArg );

	TCLAP::UnlabeledValueArg<std::string> outFilenameArg( "output", "Output image filename", false, "", "filename2" );
	cmd.add( outFilenameArg );

	cmd.parse( argc, argv );

	/***************************************************/

	std::string operatorName = operatorTypeArg.getValue();
	operatorName = boost::to_upper_copy( operatorName );
	
	boost::filesystem::path inFilename = inFilenameArg.getValue();
	boost::filesystem::path outFilename;
	if ( outFilenameArg.isSet() ) {
		outFilename = outFilenameArg.getValue();
	} else {
		outFilename = inFilename.stem() + "_" + operatorName + inFilename.extension();
	}

	std::cout << "Loading file '" << inFilename << "' ..."; std::cout.flush();
	M4D::Imaging::AImage::Ptr image = 
			M4D::Imaging::ImageFactory::LoadDumpedImage( inFilename.string() );
	std::cout << "Done\n";


	std::cout << "Initializing..."; std::cout.flush();
	std::cout << "Done\n";
	/*---------------------------------------------------------------------*/
	if ( operatorName == "SOBEL" ) {
		double threshold = thresholdArg.getValue();
		NUMERIC_TYPE_TEMPLATE_SWITCH_MACRO( image->GetElementTypeID(),
			typedef M4D::Imaging::Image< TTYPE, 3 > IMAGE_TYPE;
			IMAGE_TYPE::Ptr typedImage = IMAGE_TYPE::Cast( image );
			IMAGE_TYPE::Ptr outputImage = ImageFactory::CreateEmptyImageFromExtents< TTYPE, 3 >( typedImage->GetMinimum(), typedImage->GetMaximum(), typedImage->GetElementExtents() );

			Sobel3D( typedImage->GetRegion(), outputImage->GetRegion(), static_cast< TTYPE >( threshold ) );
			std::cout << "Saving file '" << outFilename << "' ..."; std::cout.flush();
			M4D::Imaging::ImageFactory::DumpImage( outFilename.string(), *outputImage );
			std::cout << "Done\n";
		);
	} else if ( operatorName == "THRESHOLD" ) {
		
		NUMERIC_TYPE_TEMPLATE_SWITCH_MACRO( image->GetElementTypeID(),
			typedef M4D::Imaging::Image< TTYPE, 3 > IMAGE_TYPE;
			IMAGE_TYPE::Ptr typedImage = IMAGE_TYPE::Cast( image );
			M4D::Imaging::Mask3D::Ptr outputImage = ImageFactory::CreateEmptyImageFromExtents< uint8, 3 >( typedImage->GetMinimum(), typedImage->GetMaximum(), typedImage->GetElementExtents() );

			TTYPE threshold = TypeTraits<TTYPE>::Min;
			TTYPE belowThreshold = TypeTraits<TTYPE>::Max;
			if (thresholdArg.isSet() ) {
				threshold = TTYPE( thresholdArg.getValue() );
			}
			if (belowThresholdArg.isSet() ) {
				belowThreshold = TTYPE( belowThresholdArg.getValue() );
			}
			
			thresholding3D( typedImage->GetRegion(), outputImage->GetRegion(), static_cast< TTYPE >( threshold ), static_cast< TTYPE >( belowThreshold ) );
			std::cout << "Saving file '" << outFilename << "' ..."; std::cout.flush();
			M4D::Imaging::ImageFactory::DumpImage( outFilename.string(), *outputImage );
			std::cout << "Done\n";
		);
	} else if ( operatorName == "MIN" ) {
		NUMERIC_TYPE_TEMPLATE_SWITCH_MACRO( image->GetElementTypeID(),
			TTYPE threshold = TypeTraits<TTYPE>::Max;
			if (thresholdArg.isSet() ) {
				threshold = TTYPE( thresholdArg.getValue() );
			}
			typedef M4D::Imaging::Image< TTYPE, 3 > IMAGE_TYPE;
			IMAGE_TYPE::Ptr typedImage = IMAGE_TYPE::Cast( image );
			M4D::Imaging::Mask3D::Ptr outputImage = ImageFactory::CreateEmptyImageFromExtents< uint8, 3 >( typedImage->GetMinimum(), typedImage->GetMaximum(), typedImage->GetElementExtents() );
			//M4D::Imaging::Image< uint32, 3 >::Ptr outputImage = ImageFactory::CreateEmptyImageFromExtents< uint32, 3 >( typedImage->GetMinimum(), typedImage->GetMaximum(), typedImage->GetElementExtents() );

			/*std::cout << "Finding local minima ..."; std::cout.flush();
			LocalMinimaRegions3D( typedImage->GetRegion(), outputImage->GetRegion(), threshold );
			std::cout << "Done\n";*/
			LocalMinima3D( typedImage->GetRegion(), outputImage->GetRegion(), threshold );
			std::cout << "Saving file '" << outFilename << "' ..."; std::cout.flush();
			M4D::Imaging::ImageFactory::DumpImage( outFilename.string(), *outputImage );
			std::cout << "Done\n";
		);
	} else if ( operatorName == "BORDERS" ) {
		NUMERIC_TYPE_TEMPLATE_SWITCH_MACRO( image->GetElementTypeID(),
			typedef M4D::Imaging::Image< TTYPE, 3 > IMAGE_TYPE;
			IMAGE_TYPE::Ptr typedImage = IMAGE_TYPE::Cast( image );
			M4D::Imaging::Mask3D::Ptr outputImage = ImageFactory::CreateEmptyImageFromExtents< uint8, 3 >( typedImage->GetMinimum(), typedImage->GetMaximum(), typedImage->GetElementExtents() );

			RegionBorderDetection3D( typedImage->GetRegion(), outputImage->GetRegion() );
			std::cout << "Saving file '" << outFilename << "' ..."; std::cout.flush();
			M4D::Imaging::ImageFactory::DumpImage( outFilename.string(), *outputImage );
			std::cout << "Done\n";
		);
	} else if ( operatorName == "CCL" ) {
		M4D::Imaging::Mask3D::Ptr inImage = M4D::Imaging::Mask3D::Cast( image );
		M4D::Imaging::Image< uint32, 3 >::Ptr outputImage = ImageFactory::CreateEmptyImageFromExtents< uint32, 3 >( inImage->GetMinimum(), inImage->GetMaximum(), inImage->GetElementExtents() );
		ConnectedComponentLabeling3D( inImage->GetRegion(), outputImage->GetRegion() );
		std::cout << "Saving file '" << outFilename << "' ..."; std::cout.flush();
		M4D::Imaging::ImageFactory::DumpImage( outFilename.string(), *outputImage );
		std::cout << "Done\n";
	} else if ( operatorName == "WSHED" || operatorName == "WATERSHED") {
		NUMERIC_TYPE_TEMPLATE_SWITCH_MACRO( image->GetElementTypeID(),
			TTYPE threshold = TypeTraits<TTYPE>::Max;
			if (thresholdArg.isSet() ) {
				threshold = TTYPE( thresholdArg.getValue() );
			}
			typedef M4D::Imaging::Image< TTYPE, 3 > IMAGE_TYPE;
			IMAGE_TYPE::Ptr typedImage = IMAGE_TYPE::Cast( image );
			M4D::Imaging::Mask3D::Ptr maskImage = ImageFactory::CreateEmptyImageFromExtents< uint8, 3 >( typedImage->GetMinimum(), typedImage->GetMaximum(), typedImage->GetElementExtents() );
			M4D::Imaging::Image< uint32, 3 >::Ptr labelImage = ImageFactory::CreateEmptyImageFromExtents< uint32, 3 >( typedImage->GetMinimum(), typedImage->GetMaximum(), typedImage->GetElementExtents() );
			
			std::cout << "Finding local minima ..."; std::cout.flush();
			LocalMinima3D( typedImage->GetRegion(), maskImage->GetRegion(), threshold );
			std::cout << "Done\n";

			std::cout << "Connected component labeling ..."; std::cout.flush();
			ConnectedComponentLabeling3D( maskImage->GetRegion(), labelImage->GetRegion() );
			std::cout << "Done\n";

			std::cout << "Watershed transformation ..."; std::cout.flush();
			WatershedTransformation3D( labelImage->GetRegion(), typedImage->GetRegion(), labelImage->GetRegion() );
			std::cout << "Done\n";

			std::cout << "Saving file '" << outFilename << "' ..."; std::cout.flush();
			M4D::Imaging::ImageFactory::DumpImage( outFilename.string(), *labelImage );
			std::cout << "Done\n";
		);
	} else if ( operatorName == "WSHED2" || operatorName == "WATERSHED2") {
		NUMERIC_TYPE_TEMPLATE_SWITCH_MACRO( image->GetElementTypeID(),
			TTYPE threshold = TypeTraits<TTYPE>::Max;
			if (thresholdArg.isSet() ) {
				threshold = TTYPE( thresholdArg.getValue() );
			}
			typedef M4D::Imaging::Image< TTYPE, 3 > IMAGE_TYPE;
			IMAGE_TYPE::Ptr typedImage = IMAGE_TYPE::Cast( image );
			M4D::Imaging::Image< uint32, 3 >::Ptr labelImage = ImageFactory::CreateEmptyImageFromExtents< uint32, 3 >( typedImage->GetMinimum(), typedImage->GetMaximum(), typedImage->GetElementExtents() );
			
			std::cout << "Finding local minima ..."; std::cout.flush();
			LocalMinimaRegions3D( typedImage->GetRegion(), labelImage->GetRegion(), threshold );
			std::cout << "Done\n";

			std::cout << "Watershed transformation ..."; std::cout.flush();
			WatershedTransformation3D( labelImage->GetRegion(), typedImage->GetRegion(), labelImage->GetRegion() );
			std::cout << "Done\n";

			std::cout << "Saving file '" << outFilename << "' ..."; std::cout.flush();
			M4D::Imaging::ImageFactory::DumpImage( outFilename.string(), *labelImage );
			std::cout << "Done\n";
			
			std::cout << "Adjacency graph ..."; std::cout.flush();
			pushRelabelMaxFlow( labelImage->GetRegion(), typedImage->GetRegion() );
			/*WeightedUndirectedGraph graph;
			createAdjacencyGraph( graph, labelImage->GetRegion(), typedImage->GetRegion() );
			computeMinCut( graph );*/
			std::cout << "Done\n";
		);
	} else if ( operatorName == "WSHED2CPU") {
		NUMERIC_TYPE_TEMPLATE_SWITCH_MACRO( image->GetElementTypeID(),
			TTYPE threshold = TypeTraits<TTYPE>::Max;
			if (thresholdArg.isSet() ) {
				threshold = TTYPE( thresholdArg.getValue() );
			}
			typedef M4D::Imaging::Image< TTYPE, 3 > IMAGE_TYPE;
			IMAGE_TYPE::Ptr typedImage = IMAGE_TYPE::Cast( image );
			M4D::Imaging::Image< uint32, 3 >::Ptr labelImage = ImageFactory::CreateEmptyImageFromExtents< uint32, 3 >( typedImage->GetMinimum(), typedImage->GetMaximum(), typedImage->GetElementExtents() );
			
			std::cout << "Finding local minima ..."; std::cout.flush();
			LocalMinimaRegions3D( typedImage->GetRegion(), labelImage->GetRegion(), threshold );
			std::cout << "Done\n";

			std::cout << "Watershed transformation ..."; std::cout.flush();
			WatershedTransformation3D( labelImage->GetRegion(), typedImage->GetRegion(), labelImage->GetRegion() );
			std::cout << "Done\n";

			std::cout << "Saving file '" << outFilename << "' ..."; std::cout.flush();
			M4D::Imaging::ImageFactory::DumpImage( outFilename.string(), *labelImage );
			std::cout << "Done\n";
			
			std::cout << "Adjacency graph ..."; std::cout.flush();
			//pushRelabelMaxFlow( labelImage->GetRegion(), typedImage->GetRegion() );
			WeightedUndirectedGraph graph;
			createAdjacencyGraph( graph, labelImage->GetRegion(), typedImage->GetRegion() );
			computeMinCut( graph );
			std::cout << "Done\n";
		);
	} else {
		std::cout << operatorName << " is not valid operator";
		return 1;
	}

	/*---------------------------------------------------------------------*/

	return 0;
}

