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
