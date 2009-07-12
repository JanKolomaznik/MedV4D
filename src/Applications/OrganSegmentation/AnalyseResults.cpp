#include "AnalyseResults.h"
#include "common/Common.h"

void
AnalyseResults( const InputImageType &image, const M4D::Imaging::Mask3D &mask, AnalysisRecord &record )
{
	//Test if image and mask have the same size
	ASSERT( image.GetSize() == mask.GetSize() );


	InputImageType::Iterator it1 = image.GetIterator();
	M4D::Imaging::Mask3D::Iterator it2 = mask.GetIterator();

	int64 voxelCount = 0;
	float32 elementVolume = VectorCoordinateProduct( mask.GetElementExtents() );

	while( (!it1.IsEnd()) && (!it2.IsEnd()) ) {
		
		if( *it2 ) {
			++voxelCount;
		} else {

		}

		++it1;
		++it2;
	}

	record.organVolume = voxelCount * elementVolume;
	LOG( "Voxel count = " << voxelCount );
	LOG( "Organ volume = " << voxelCount << " * " << elementVolume << " = " << record.organVolume );

}
