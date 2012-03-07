#ifndef WATERSHED_TRANSFORMATION_H
#define WATERSHED_TRANSFORMATION_H

template< typename InRegionType >
void
RegionBorderDetection3D( InRegionType input,  M4D::Imaging::MaskRegion3D output );

template< typename TEType >
void
WatershedTransformation3D( 
			M4D::Imaging::ImageRegion< uint32, 3 > aLabeledMarkerRegions, 
			M4D::Imaging::ImageRegion< TEType, 3 > aInput, 
			M4D::Imaging::ImageRegion< uint32, 3 > aOutput );

#endif //WATERSHED_TRANSFORMATION_H