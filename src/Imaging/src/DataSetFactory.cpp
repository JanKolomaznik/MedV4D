
#include "common/Common.h"

#include "../DataSetFactory.h"
#include "../dataSetClassEnum.h"
#include "../ImageFactory.h"


using namespace M4D::ErrorHandling;
using namespace M4D::Imaging;
using namespace M4D::IO;

AbstractDataSet::Ptr
DataSetFactory::DeserializeDataset(InStream &stream)
{
	uint32 startMAGIC = 0;
	uint32 datasetType = 0;
	uint32 formatVersion = 0;

	//Read stream header
	stream.Get<uint32>( startMAGIC );
	if( startMAGIC != DUMP_START_MAGIC_NUMBER ) {
		_THROW_ EWrongStreamBeginning();
	}
	
	stream.Get<uint32>( formatVersion );
	if( formatVersion != ACTUAL_FORMAT_VERSION ) {
		_THROW_ EWrongFormatVersion();
	}

	stream.Get<uint32>( datasetType );
		
	// main switch acording data set type
	switch((DataSetType) datasetType )
	{
	case DATASET_IMAGE:
		D_PRINT("D-Set factory: Creating Image");
		return DeserializeImageFromStream( stream );
		break;
		
	default:
		ASSERT(false);
	}
	return AbstractDataSet::Ptr();
}

void 
DataSetFactory::DeserializeDataset(M4D::IO::InStream &stream, AbstractDataSet &dataset)
{
	switch( dataset.GetDatasetType() )
	{
	case DATASET_IMAGE:
		return DeserializeImage( stream, static_cast<AbstractImage &>( dataset ) );
		break;
		
	default:
		ASSERT(false);
	}

}
void 
DataSetFactory::SerializeDataset(M4D::IO::OutStream &stream, const AbstractDataSet &dataset)
{
	switch( dataset.GetDatasetType() )
	{
	case DATASET_IMAGE:
		return SerializeImage( stream, static_cast<const AbstractImage &>( dataset ) );
		break;
		
	default:
		ASSERT(false);
	}

}

/*AbstractDataSet::Ptr
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
			    		ds = ImageFactory::DeserializeImage< TTYPE, DIM >(stream) )
			  );
	
	// deserialize data
	ds->DeSerializeData(stream);
	
	return ds;
}*/
