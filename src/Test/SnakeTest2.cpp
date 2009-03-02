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
typedef	Image< uint8, 2 > ImageType;

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

typedef RegionImageEnergy< Curve, ImageRegionType, Distribution > RegionEnergy;
//typedef RegionEnergy FinalEnergy;

typedef SegmentationEnergy< Curve, RegionEnergy, InternalCurveEnergy< Curve >, DummyEnergy3 > FinalEnergy;

typedef EnergicSnake< Curve, FinalEnergy > Snake;


int
main( int argc, char **argv )
{

	Snake snake;

	SET_DOUT( std::cerr );
	std::ofstream file( "gradients.txt" );

	string inFilename1 = "Circle.dump";
	//string inFilename2 = "CircleSmoothLaplace.dump";

	//std::cout << "Loading file..."; std::cout.flush();
	M4D::Imaging::AbstractImage::Ptr aimage = 
			M4D::Imaging::ImageFactory::LoadDumpedImage( inFilename1 );

	ImageType::Ptr image = ImageType::CastAbstractImage( aimage );

	//aimage = M4D::Imaging::ImageFactory::LoadDumpedImage( inFilename2 );

	//GradientImageType::Ptr gradientImage = GradientImageType::CastAbstractImage( aimage );

	//std::cout << "Done\n";

	Curve curve;
	//Add points
	
	float radius = 20.0f;
	Coords center = Coords(57,127);
	int segments = 16;
	float angle = -2*PI / (float)segments;

	for( int i = 0; i < segments; ++i ) {
		Coords np = center + (radius * Coords(sin(angle*i), cos(angle*i)) );
		curve.AddPoint( np );
	}

	/*curve.AddPoint( Coords(50,50) );
	curve.AddPoint( Coords(200,50) );
	curve.AddPoint( Coords(135,200) );
	curve.AddPoint( Coords(50,70) );*/


	curve.SetCyclic();
	curve.Sample( 5 );

	snake.Initialize( curve );
	snake.SetGamma( 1.0f );
	snake.SetImageEnergyBalance( 1.0f );
	snake.SetInternalEnergyBalance( 0.5f );
	snake.SetConstrainEnergyBalance( 0.0f );
	snake.SetRegionStatRegion( image->GetRegion() );
	//snake.GetEnergyModel().SetRegion2( gradientImage->GetRegion() );

	unsigned i = 0;
	while( i < 40 ) {
		i = snake.Step();
		if( i % 4 == 0 ) 
		{
			PrintCurve( cout, snake.GetCurrentCurve() );
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
