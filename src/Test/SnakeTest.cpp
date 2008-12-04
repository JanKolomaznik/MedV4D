#include "iostream"
#include "Imaging/BSpline.h"
#include <fstream>
#include "Imaging/EnergicSnake.h"

using namespace std;

using namespace M4D;
using namespace M4D::Imaging::Geometry;
using namespace M4D::Imaging::Algorithms;

typedef float CoordType;
typedef BSpline< CoordType, 2 > Curve;
typedef PointSet< CoordType, 2 > 	Points;
typedef Coordinates< CoordType, 2 > Coords;

typedef EnergicSnake< Curve, EFConvergeToPoint< Curve > > Snake;


void
SetLine( const Coords &a, const Coords &b )
{
	//cout <<	"set arrow from " << a[0] << "," << a[1] 
	//	<< " to " << b[0] << "," << b[1] << "\n";
	
	cout << a[0] << " " << a[1]  << "\n";
		
}

Snake snake;

int
main( int argc, char **argv )
{
	SET_DOUT( std::cerr );
	std::ofstream file( "gradients.txt" );

	Curve curve;
	//Add points
	
	curve.AddPoint( Coords(10,10) );
	curve.AddPoint( Coords(120,220) );
	curve.AddPoint( Coords(250,10) );
	curve.AddPoint( Coords(210,130) );
	curve.AddPoint( Coords(250,245) );
	curve.AddPoint( Coords(140,180) );
	curve.AddPoint( Coords(10,230) );
	curve.AddPoint( Coords(45,100) );


	curve.SetCyclic();
	//Sample
	//curve.Sample( 5 );

	snake.Initialize( curve );
	snake.GetEnergyModel().SetCenterPoint( Coords(128,128) );

	while( !snake.Converged() ) {
		unsigned i = snake.Step();
		if( i % 3 == 0 ) {
			PrintCurve( cout, snake.GetCurrentCurve() );
			cout << endl;
			cout << endl;
		}
	}
	/*for( unsigned i = 0; i < 120; ++i ) {
		snake.Step();
		if( i % 3 == 0 ) {
			PrintCurve( cout, snake.GetCurrentCurve() );
			cout << endl;
			cout << endl;
		}
	}*/

	return 0;
}
