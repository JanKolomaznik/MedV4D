#include "iostream"
#include "Common.h"
#include "Imaging/BSpline.h"
#include "Imaging/Image.h"
#include "Imaging/ImageFactory.h"
#include <fstream>
#include "Imaging/EnergicSnake.h"
#include "Imaging/EnergyModels.h"
#include "Imaging/GeometricAlgorithms.h"

using namespace std;

using namespace M4D;
using namespace M4D::Imaging;
using namespace M4D::Imaging::Geometry;
using namespace M4D::Imaging::Algorithms;

typedef float CoordType;
typedef BSpline< CoordType, 2 > Curve;
typedef PointSet< CoordType, 2 > 	Points;
typedef Vector< CoordType, 2 > Coords;

typedef	Image< int16, 2 > GradientImageType;
typedef	Image< int16, 2 > ImageType;

typedef GradientImageType::SubRegion GradientRegionType;
typedef ImageType::SubRegion ImageRegionType;

/*typedef SimpleBaloonForce< Curve > BaloonEnergy;
typedef GradientMagnitudeEnergy< Curve, GradientRegionType > GradientEnergy;
typedef DoubleEnergyFunctional< Curve, BaloonEnergy, GradientEnergy > FinalEnergy;*/

//typedef UnifiedImageEnergy< Curve, ImageRegionType, GradientRegionType > FinalEnergy;
class Distribution
{
public:
	float32
	LogProbabilityRatio( int val )
	{ 
		float32 P1 = ((float32)(255-val))/255.0f + 0.0001;
		float32 P2 = ((float32)val)/255.0f  + 0.0001;
		return log( P1 / P2 );
	}
};

//typedef RegionImageEnergy< Curve, ImageRegionType, Distribution > ImageEnergy;
typedef UnifiedImageEnergy< Curve, ImageRegionType, GradientRegionType, Distribution > ImageEnergy;
//typedef RegionEnergy FinalEnergy;

typedef SegmentationEnergy< Curve, ImageEnergy, InternalCurveEnergy< Curve >, DummyEnergy3 > FinalEnergy;
//typedef SegmentationEnergy< Curve, RegionEnergy, InternalCurveEnergy< Curve >, DummyEnergy3 > FinalEnergy;

typedef EnergicSnake< Curve, FinalEnergy > Snake;


int
main( int argc, char **argv )
{

	Snake snake;

	SET_DOUT( std::cerr );
	SET_LOUT( std::cerr );
	std::ofstream file( "gradients.txt" );

	string inFilename1 = "Circle.dump";
	string inFilename2 = "CircleSmoothLaplace.dump";

	//std::cout << "Loading file..."; std::cout.flush();
	M4D::Imaging::AbstractImage::Ptr aimage = 
			M4D::Imaging::ImageFactory::LoadDumpedImage( inFilename1 );

	ImageType::Ptr image = ImageType::CastAbstractImage( aimage );

	aimage = M4D::Imaging::ImageFactory::LoadDumpedImage( inFilename2 );

	GradientImageType::Ptr gradientImage = GradientImageType::CastAbstractImage( aimage );

	//std::cout << "Done\n";

	/*GradientImageType::Iterator it = gradientImage->GetIterator();
	float32 value = *it;
	while( !it.IsEnd() ) {
		if( Abs(*it) > value ) {
			value = *it;
		}
		++it;
	}*/
	//std::cerr << " LLLLLL "<< gradientImage->GetElement( Vector<int32,2>(182, 126) ) << "\n";
	ImageFactory::DumpImage( "testing.dump", *gradientImage );

	Curve curve;
	//Add points
	
	float radius = 40.0f;
	Coords center = Coords(90,127);
	int segments = 8;
	float angle = -2*PI / (float)segments;

	/*for( int i = 0; i < segments; ++i ) {
		Coords np = center + (radius * Coords(sin(angle*i), cos(angle*i)) );
		curve.AddPoint( np );
	}*/

	
	curve.AddPoint( Coords(50,110) );
	curve.AddPoint( Coords(230,50) );
	curve.AddPoint( Coords(230,200) );
	curve.AddPoint( Coords(80,150) );
	curve.AddPoint( Coords(50,130) );

	curve.Scale( Vector<float32,2>(0.3f, 0.3f), Vector<float32,2>( 125, 125 ) );
	

	curve.SetCyclic();
	curve.Sample( 5 );

	snake.Initialize( curve );
	snake.SetGamma( 1.0f );
	snake.SetAlpha( 0.0f );
	snake.SetImageEnergyBalance( 0.0f );
	snake.SetInternalEnergyBalance( 1.0f );
	snake.SetConstrainEnergyBalance( 0.0f );
	snake.SetRegion1( image->GetRegion() );
	snake.SetRegion2( gradientImage->GetRegion() );
	//snake.SetRegionStatRegion( image->GetRegion() );

	snake.SetSelfIntersectionTestPeriod( 0 );
	snake.SetSegmentLengthsTestPeriod( 0 );

	unsigned i = 0;
	while( i < 30 ) {
		i = snake.Step();
		if( i % 4 == 0 ) 
		{
			PrintCurve( cout, snake.GetCurrentCurve() );
			//PrintPointSet( cout, snake.GetCurrentCurve() );
			cout << endl;
			cout << endl;
		}
	}
	PrintCurve( cout, snake.GetCurrentCurve() );
			cout << endl;
			cout << endl;
	PrintPointSet( cout, snake.GetCurrentCurve() );

	return 0;
}
