
#include "MedV4D/Common/Common.h"

#include "MedV4D/Imaging/DatasetFactory.h"
#include "MedV4D/Imaging/DatasetClassEnum.h"
#include "MedV4D/Imaging/ImageFactory.h"


using namespace M4D::ErrorHandling;
using namespace M4D::Imaging;
using namespace M4D::IO;

ADataset::Ptr
DatasetFactory::DeserializeDataset ( InStream &stream )
{
        uint32 datasetType = 0;


        //Read stream header
        datasetType = DeserializeHeader ( stream );

        // main switch acording data set type
        switch ( ( DatasetType ) datasetType ) {
        case DATASET_IMAGE:
                LOG ( "Dataset factory: Deserializing image from stream." );
                return DeserializeImageFromStream ( stream );
                break;
        case DATASET_SLICED_GEOMETRY:
                LOG ( "Dataset factory: Deserializing sliced geometry from stream." );
                return DeserializeSlicedGeometryFromStream ( stream );
                break;
        default:
                ASSERT ( false );
        }
        return ADataset::Ptr();
}

void
DatasetFactory::DeserializeDataset ( M4D::IO::InStream &stream, ADataset &dataset )
{
        switch ( dataset.GetDatasetType() ) {
        case DATASET_IMAGE:
                LOG ( "Dataset factory: Deserializing image from stream to prepared dataset." );
                return DeserializeImage ( stream, AImage::Cast ( dataset ) );
                break;
        case DATASET_SLICED_GEOMETRY:
                LOG ( "Dataset factory: Deserializing sliced geometry from stream to prepared dataset." );
                return DeserializeSlicedGeometryFromStream ( stream, ASlicedGeometry::Cast ( dataset ) );
                break;
        default:
                ASSERT ( false );
        }

}
void
DatasetFactory::SerializeDataset ( M4D::IO::OutStream &stream, const ADataset &dataset )
{
        switch ( dataset.GetDatasetType() ) {
        case DATASET_IMAGE:
                LOG ( "Dataset factory: Serialize image." );
                return SerializeImage ( stream, AImage::Cast ( dataset ) );
                break;
        case DATASET_SLICED_GEOMETRY:
                LOG ( "Dataset factory: Serialize sliced geometry." );
                return SerializeSlicedGeometry ( stream, ASlicedGeometry::Cast ( dataset ) );
                break;
        default:
                ASSERT ( false );
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
