
#include "Common.h"

#include "../DataSetFactory.h"
#include "../dataSetClassEnum.h"
#include "../ImageFactory.h"


using namespace M4D::ErrorHandling;
using namespace M4D::Imaging;

AbstractDataSet::ADataSetPtr
DataSetFactory::CreateDataSet(iAccessStream &stream)
{
	uint8 dsType;
	stream >> (uint8&) dsType;	// read
	
	// main switch acording data set type
	switch((DataSetType) dsType)
	{
	case DATASET_IMAGE:
		return CreateImage(stream);
		break;
		
	default:
		ASSERT(false);
	}
	//return 1;	// program shall never go here
	return AbstractDataSet::ADataSetPtr();
}

AbstractDataSet::ADataSetPtr
DataSetFactory::CreateImage(iAccessStream &stream)
{
	AbstractDataSet::ADataSetPtr ds;
	
	uint16 dim, elemType;
	stream >> elemType >> dim;   // get class properties
	
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
