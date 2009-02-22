
#include "Common.h"

#include "../DataSetFactory.h"
#include "../dataSetClassEnum.h"
#include "../ImageFactory.h"


using namespace M4D::ErrorHandling;
using namespace M4D::Imaging;

AbstractDataSet::Ptr
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
	return AbstractDataSet::Ptr();
}

AbstractDataSet::Ptr
DataSetFactory::CreateImage(iAccessStream &stream)
{
	AbstractDataSet::Ptr ds;
	
	uint16 dim, elemType;
	stream >> elemType >> dim;   // get class properties

	// create approp class
	NUMERIC_TYPE_TEMPLATE_SWITCH_MACRO( elemType, 
			    DIMENSION_TEMPLATE_SWITCH_MACRO( dim, 
			    		ds = Image< TTYPE, DIM >::Ptr(new Image< TTYPE, DIM >()) )
			  );
	
	ds->DeSerializeProperties(stream);
	
	ImageFactory::AllocateDataAccordingProperties(ds);
	
	return ds;
}
