
#include "Common.h"

#include "../DataSetFactory.h"
#include "../dataSetClassEnum.h"
#include "../ImageFactory.h"


using namespace M4D::ErrorHandling;
using namespace M4D::Imaging;

AbstractDataSet::ADataSetPtr
DataSetFactory::CreateDataSet(iAccessStream &stream)
{
	Endianness endian;
	DataSetType dsType;
	
	stream >> (uint8&) endian >> (uint8&) dsType;	// read
	
	// main switch acording data set type
	switch(dsType)
	{
	case DATASET_IMAGE:
		CreateImage(stream);
		break;
		
	case DATASET_TRIANGLE_MESH:
		break;
				
	default:
		ASSERT(false);
	}
	return null;
}

AbstractDataSet::ADataSetPtr
DataSetFactory::CreateImage(iAccessStream &stream)
{
	AbstractDataSet::ADataSetPtr ds;
	
	uint8 dim, elemType;
	stream >> dim >> elemType;   // get class properties
	
	int32 minimums[ dim ];
	int32 maximums[ dim ];
	float32 elExtents[ dim ];

	for( unsigned i = 0; i < dim; ++i ) {

		stream >> minimums[ i ];
		stream >> maximums[ i ];
		stream >> elExtents[ i ];
	}

	// create approp class
	NUMERIC_TYPE_TEMPLATE_SWITCH_MACRO( 
			elemType,
			ds = ImageFactory::CreateEmptyImageFromExtents< TTYPE >( 
					dim, minimums, maximums, elExtents )				
	);

	ds.get()->DeSerialize(stream);
	
	return ds;
}