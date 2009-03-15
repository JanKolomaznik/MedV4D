#include "Imaging.h"

ImageType
MaskType
DiscreteHistogram
Transformation
GridType

void
GetPoles( const MaskType & mask, MaskType::PointType &north, MaskType::PointType &south );

Transformation
GetTransformation( MaskType::PointType north, MaskType::PointType south, MaskType::ElementExtentsType elementExtents ); 

void
FillInOutHistograms( DiscreteHistogram &inHistogram, DiscreteHistogram &outHistogram, const ImageType &image, const MaskType &mask );

void
FillGrid( Transformation tr, const ImageType &image, const MaskType &mask, GridType &grid );

void
IncorporateGrids( const GridType &last, GridType &general );

void
TrainingStep( 
		DiscreteHistogram	&generalInHistogram, 
		DiscreteHistogram	&generalOutHistogram,
		const ImageType		&image, 
		const MaskType		&mask,
		GridType		&generalGrid
	    );
