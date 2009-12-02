
#include "common/Common.h"

#include "Imaging/DatasetFactory.h"
#include "Imaging/DatasetClassEnum.h"
#include "Imaging/ImageFactory.h"


using namespace M4D::ErrorHandling;
using namespace M4D::Imaging;
using namespace M4D::IO;

ADataset::Ptr
DatasetFactory::DeserializeDataset(InStream &stream)
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
	switch((DatasetType) datasetType )
	{
	case DATASET_IMAGE:
		D_PRINT("D-Set factory: Creating Image");
		return DeserializeImageFromStream( stream );
		break;
		
	default:
		ASSERT(false);
	}
	return ADataset::Ptr();
}

void 
DatasetFactory::DeserializeDataset(M4D::IO::InStream &stream, ADataset &dataset)
{
	switch( dataset.GetDatasetType() )
	{
	case DATASET_IMAGE:
		return DeserializeImage( stream, static_cast<AImage &>( dataset ) );
		break;
		
	default:
		ASSERT(false);
	}

}
void 
DatasetFactory::SerializeDataset(M4D::IO::OutStream &stream, const ADataset &dataset)
{
	switch( dataset.GetDatasetType() )
	{
	case DATASET_IMAGE:
		return SerializeImage( stream, static_cast<const AImage &>( dataset ) );
		break;
		
	default:
		ASSERT(false);
	}

}

/*ADataset::Ptr
DatasetFactory::CreateImage(InStream &stream)
{
	ADataset::Ptr ds;
	
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
