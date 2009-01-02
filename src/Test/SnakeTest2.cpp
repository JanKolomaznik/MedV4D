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
typedef Coordinates< CoordType, 2 > Coords;

typedef	Image< int16, 2 > GradientImageType;
typedef	Image< uint8, 2 > ImageType;

typedef GradientImageType::SubRegion GradientRegionType;
typedef ImageType::SubRegion ImageRegionType;

/*typedef SimpleBaloonForce< Curve > BaloonEnergy;
typedef GradientMagnitudeEnergy< Curve, GradientRegionType > GradientEnergy;
typedef DoubleEnergyFunctional< Curve, BaloonEnergy, GradientEnergy > FinalEnergy;*/

typedef UnifiedImageEnergy< Curve, ImageRegionType, GradientRegionType > FinalEnergy;

typedef EnergicSnake< Curve, FinalEnergy > Snake;

Snake snake;

int
main( int argc, char **argv )
{
	SET_DOUT( std::cerr );
	std::ofstream file( "gradients.txt" );

	string inFilename1 = "Circle.dump";
	string inFilename2 = "CircleSmoothLaplace.dump";

	//std::cout << "Loading file..."; std::cout.flush();
	M4D::Imaging::AbstractImage::AImagePtr aimage = 
			M4D::Imaging::ImageFactory::LoadDumpedImage( inFilename1 );

	ImageType::Ptr image = ImageType::CastAbstractImage( aimage );

	aimage = M4D::Imaging::ImageFactory::LoadDumpedImage( inFilename2 );

	GradientImageType::Ptr gradientImage = GradientImageType::CastAbstractImage( aimage );

	//std::cout << "Done\n";

	Curve curve;
	//Add points
	
	float radius = 30.0f;
	Coords center = Coords(127,127);
	int segments = 16;
	float angle = -2*PI / (float)segments;

	for( int i = 0; i < segments; ++i ) {
		Coords np = center + (radius * Coords(sin(angle*i), cos(angle*i)) );
		curve.AddPoint( np );
	}

	/*curve.AddPoint( Coords(120,120) );
	curve.AddPoint( Coords(135,120) );
	curve.AddPoint( Coords(135,135) );
	curve.AddPoint( Coords(120,135) );*/

	
/*	curve.AddPoint( Coords(30,30) );
	curve.AddPoint( Coords(225,30) );
	curve.AddPoint( Coords(225,225) );
	curve.AddPoint( Coords(30,225) );*/
	/* 
	curve.AddPoint( Coords(127,74) );
	curve.AddPoint( Coords(182,127) );
	curve.AddPoint( Coords(127,182) );
	curve.AddPoint( Coords(74,127) );
	curve.AddPoint( Coords(127,74) );
	curve.AddPoint( Coords(182,127) );
	curve.AddPoint( Coords(127,182) );
	curve.AddPoint( Coords(74,127) );*/

	curve.SetCyclic();
	curve.Sample( 5 );
	snake.Initialize( curve );
	snake.GetEnergyModel().SetRegion1( image->GetRegion() );
	snake.GetEnergyModel().SetRegion2( gradientImage->GetRegion() );
/*	snake.GetEnergyModel().GetSecondModel().SetRegion( image->GetRegion() );
	snake.GetEnergyModel().SetAlpha(0.1);
	snake.GetEnergyModel().SetBeta(25.0);*/

	while( !snake.Converged() ) {
		unsigned i = snake.Step();
		if( i % 3 == 0 ) 
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

	bool result;
	result = LineIntersectionTest( Coords( -1,-1 ), Coords( 10,10 ), Coords( -6,-5 ), Coords( 6,5 ) );
	cerr << "R1 = " << result << endl;
	result = LineIntersectionTest( Coords( 0,0 ), Coords( 10,10 ), Coords( 0,0 ), Coords( 10,7 ) );
	cerr << "R2 = " << result << endl;
	return 0;
}
