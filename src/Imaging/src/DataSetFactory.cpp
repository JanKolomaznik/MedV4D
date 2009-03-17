
#include "Common.h"

#include "../DataSetFactory.h"
#include "../dataSetClassEnum.h"
#include "../ImageFactory.h"


using namespace M4D::ErrorHandling;
using namespace M4D::Imaging;
using namespace M4D::IO;

AbstractDataSet::Ptr
DataSetFactory::CreateDataSet(InStream &stream)
{
	uint8 dsType;
	stream.Get<uint8>(dsType);
	
	// main switch acording data set type
	switch((DataSetType) dsType)
	{
	case DATASET_IMAGE:
		D_PRINT("D-Set factory: Creating Image");
		return CreateImage(stream);
		break;
		
	default:
		ASSERT(false);
	}
	//return 1;	// program shall never go here
	return AbstractDataSet::Ptr();
}

AbstractDataSet::Ptr
DataSetFactory::CreateImage(InStream &stream)
{
	AbstractDataSet::Ptr ds;
	
	uint16 dim, elemType;
	stream.Get<uint16>(elemType);
	stream.Get<uint16>(dim);
	
	D_PRINT("Elemtype: " << elemType << ", dim: " << dim);

	// create approp class
	NUMERIC_TYPE_TEMPLATE_SWITCH_MACRO( elemType, 
			    DIMENSION_TEMPLATE_SWITCH_MACRO( dim, 
			    		ds = Image< TTYPE, DIM >::Ptr(new Image< TTYPE, DIM >()) )
			  );
	
	ds->DeSerializeProperties(stream);
	
	// allocate data buffer according read properties
	NUMERIC_TYPE_TEMPLATE_SWITCH_MACRO( elemType, 
		DIMENSION_TEMPLATE_SWITCH_MACRO( dim, 
			ImageFactory::AllocateDataAccordingProperties<TTYPE, DIM>(*(Image<TTYPE, DIM> *)ds.get()) )
		);
	
	// deserialize data
	ds->DeSerializeData(stream);
	
	return ds;
}
