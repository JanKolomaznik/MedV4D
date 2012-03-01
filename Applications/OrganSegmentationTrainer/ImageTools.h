#ifndef IMAGE_TOOLS_H
#define IMAGE_TOOLS_H

#include "MedV4D/Imaging/Imaging.h"
#include "MedV4D/Common/Common.h"

#include <boost/filesystem.hpp>
#include <vector>
#include <string>

typedef M4D::Imaging::Image< int16, 3 >		ImageType;
typedef M4D::Imaging::Mask3D			MaskType;
typedef M4D::Imaging::Histogram< int64 >	DiscreteHistogram;
typedef M4D::Imaging::Histogram< float32 >	FloatHistogram;

//typedef M4D::Imaging::Transformation		Transformation;
typedef M4D::Imaging::ProbabilityGrid		GridType;

typedef boost::filesystem::path			Path;


struct TrainingDataInfo
{
	Path first;
	Path second;
	Path dir;
};

typedef std::vector< TrainingDataInfo >		TrainingDataInfos;

float32
RatioLogarithm( float32 a, float32 b );

void
GetPoles( const MaskType & mask, MaskType::PointType &north, MaskType::PointType &south );

/*Transformation
GetTransformation( MaskType::PointType north, MaskType::PointType south, MaskType::ElementExtentsType elementExtents ); */

void
FillInOutHistograms( DiscreteHistogram &inHistogram, DiscreteHistogram &outHistogram, const ImageType &image, const MaskType &mask );

void
FillGrid( M4D::Imaging::Transformation tr, const ImageType &image, const MaskType &mask, GridType &grid, int32 minHist, int32 maxHist );

void
IncorporateGrids( const GridType &last, GridType &general );

void
TrainingStep( 
		DiscreteHistogram	&generalInHistogram, 
		DiscreteHistogram	&generalOutHistogram,
		const ImageType		&image, 
		const MaskType		&mask,
		GridType		&generalGrid,
		int32 			minHist,
		int32			maxHist
	    );

M4D::Imaging::CanonicalProbModel::Ptr
Train( const TrainingDataInfos &infos, Vector< uint32, 3 > size, Vector< float32, 3 > step, Vector< float32, 3 > origin, int32 minHist, int32 maxHist );

void
ConsolidateGeneralGrid( GridType &grid, int32 count );

void
GetTrainingSetInfos( const Path & dirPath, std::string indexExtension, TrainingDataInfos &infos, bool recursive = false );


#endif /*IMAGE_TOOLS_H*/
