
#include "Common.h"
#include <fstream>
#include <string>

#include "testDataGenerator.h"

using namespace std;
using namespace M4D::Imaging;


#define STD_IN_NAME "in.mv4d"

int main ( int argc, char *argv[] )
{
	std::ofstream logFile( "Log.txt" );
	SET_LOUT( logFile );

	D_COMMAND( std::ofstream debugFile( "Debug.txt" ); );
	SET_DOUT( debugFile );
	
	if(argc > 3)
		LOG("Too much of params!")
		
	if(argc > 1)
	{
		//char *in = argv[1];
	}

	M4D::Imaging::Image<uint16, 2>::Ptr im = CreateTestImage(64, 64);
	
	im->Dump();

  return 0;
}
