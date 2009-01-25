
#include "DataSetFactory.h"
#include "../dataSetClassEnum.h"
#include "Common.h"

using M4D::ErrorHandling;

AbstractDataSet::ADataSetPtr
DataSetFactory::CreateDataSet(iAccessStream &stream)
{
	Endianness endian;
	DataSetType dsType;
	
	stream >> (uint8) endian >> (uint8) dsType;	// read
	
	// main switch acording data set type
	switch(dsType)
	{
	case DATASET_IMAGE:
		break;
		
	case DATASET_TRIANGLE_MESH:
		break;
				
	default:
		throw ExceptionBase("Bad data set ID");
	}
}

AbstractDataSet::ADataSetPtr
DataSetFactory::CreateImage(iAccessStream &stream)
{
	uint8 dim, elemType;
	stream >> dim >> elemType;   // get class properties

	// create approp class
	AbstractDataSet::ADataSetPtr ds;
      NUMERIC_TYPE_TEMPLATE_SWITCH_MACRO( elemType, 
		    DIMENSION_TEMPLATE_SWITCH_MACRO( dim, 
			    ds = new Image< TTYPE, DIM >() )
		  );

	ds.DeSerialize(stream);
	
	return ds;
}