#include "ImageTools.h"

void
TrainingStep( 
		DiscreteHistogram	&generalInHistogram, 
		DiscreteHistogram	&generalOutHistogram,
		const ImageType		&image, 
		const MaskType		&mask,
		GridType		&generalGrid
	    )
{
	MaskType::PointType north;
	MaskType::PointType south;

	GetPoles( mask, north, south );

	Transformation tr = GetTransformation( north, south, Mask.GetElementExtents() );

	DiscreteHistogram inHistogram;
	SiscreteHistogram outHistogram;
	//TODO - set histograms

	FillInOutHistograms( inHistogram, outHistogram, image, mask );

	generalInHistogram += inHistogram;
	generalOutHistogram += outHistogram;

	GridType grid;
	//TODO - set grid
	
	FillGrid( tr, image, mask, grid );

	IncorporateGrids( grid, generalGrid );

}


void
FillGrid( Transformation tr, const ImageType &image, const MaskType &mask, GridType &grid )
{
	Vector< uint32, 3 > size = grid.GetSize();
	Vector< float32, 3 > step = grid.GetGridStep();
	Vector< float32, 3 > halfStep = 0.5 * step;
	Vector< float32, 3 > origin = grid.GetOrigin();

	Vector< uint32, 3 > idx;
	for( idx[0] = 0; idx[0] < size[0]; ++idx[0] ) {
		for( idx[1] = 0; idx[1] < size[1]; ++idx[1] ) {
			for( idx[2] = 0; idx[2] < size[0]; ++idx[2] ) {
	
			}
		}
	}
}

void
FillInOutHistograms( DiscreteHistogram &inHistogram, DiscreteHistogram &outHistogram, const ImageType &image, const MaskType &mask )
{
	if( image.GetMinimum() != mask.GetMinimum() ||
		image.GetMaximum() != mask.GetMaximum() ) {
		_THROW_ M4D::ErrorHandling::ExceptionBase( "Mask and image incompatible." );
	}
	ImageType::Iterator imageIt = image.GetIterator();
	MaskType::Iterator maskIt = mask.GetIterator();

	while( !imageIt.IsEnd() ) {

		if( *maskIt == 0 ) {
			outHistogram.IncCell( *imageIt );
		} else {
			inHistogram.IncCell( *imageIt );
		}

		++imageIt;
		++maskIt;
	}
}

static MaskType::SliceRegion::PointType 
FindMaskCenterOfGravity( const MaskType::SliceRegion &region )
{
	MaskType::SliceRegion::PointType sum;
	MaskType::SliceRegion::PointType idx = region.GetMinimum();
	MaskType::SliceRegion::PointType max = region.GetMaximum();
	int32 count = 0;
	for( ; idx[0] < max[0]; ++idx[0] ) {
		for( ; idx[1] < max[1]; ++idx[1] ) {
			if( region.GetElement( idx ) ) {
				++count;
				sum += idx;
			}
		}
	}
	
	if( count == 0 ) {
		_THROW_ M4D::ErrorHandling::ExceptionBase( "Center of gravity unable to find." );
	}
	return MaskType::SliceRegion::PointType( sum[0] / count, sum[1] / count, sum[2] / count );
}

void
GetPoles( const MaskType & mask, MaskType::PointType &north, MaskType::PointType &south )
{
	int32 southSliceCoord = mask.GetMinimum()[2];
	int32 northSliceCoord = mask.GetMaximum()[2]-1;
	MaskType::SliceRegion southRegion = mask.GetSlice( southSliceCoord );
	MaskType::SliceRegion northRegion = mask.GetSlice( northSliceCoord );

	MaskType::SliceRegion::PointType southTmp = FindMaskCenterOfGravity( southRegion );
	MaskType::SliceRegion::PointType northTmp = FindMaskCenterOfGravity( northRegion );

	south = MaskType::PointType( southTmp[0], southTmp[1], southSliceCoord );
	north = MaskType::PointType( northTmp[0], northTmp[1], northSliceCoord );
}


