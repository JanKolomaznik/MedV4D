
#include "Common.h"
#include <fstream>
#include <string>
#include "streams/fileStream/fileStream.h"

#include "Imaging/DataSetFactory.h"
#include "testDataGenerator.h"

using namespace std;
using namespace M4D::Imaging;
using namespace M4D::IO;

#define STD_IN_NAME "in.mv4d"
#define STD_OUT_NAME "out.mv4d"

int main ( int argc, char *argv[] )
{
	std::ofstream logFile( "Log.txt" );
	SET_LOUT( logFile );

	D_COMMAND( std::ofstream debugFile( "Debug.txt" ); );
	SET_DOUT( debugFile );
	
	if(argc > 3)
	{
		LOG("Too much of params!")
		return -1;
	}

	//M4D::Imaging::Image<uint16, 2>::Ptr im = CreateTestImage(64, 64);
	try {
		
		M4D::Imaging::AbstractDataSet::ADataSetPtr im;
		if(argc > 1)
		{
			FileStream s(argv[1], MODE_READ);
			im = DataSetFactory::CreateDataSet(s);
		}
		else 
		{
			FileStream s(STD_IN_NAME, MODE_READ);
			im = DataSetFactory::CreateDataSet(s);
		}
		
		im->Dump();
		
		if(argc > 2)
		{
			FileStream s(argv[2], MODE_WRITE);
			im->Serialize(s);
		}
		else {
			FileStream s(STD_OUT_NAME, MODE_WRITE);
			im->Serialize(s);
		}
	} catch (std::exception &e) {
		LOG(e.what());
	}

  return 0;
}
