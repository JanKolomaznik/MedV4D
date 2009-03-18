#ifndef IMAGE_TOOLS_H
#define IMAGE_TOOLS_H

#include "Imaging.h"
#include "Common.h"

#include <boost/filesystem.hpp>
#include <vector>
#include <string>

typedef M4D::Imaging::Image< int16, 3 >		ImageType;
typedef M4D::Imaging::Mask3D			MaskType;
typedef M4D::Imaging::Histogram< int64 >	DiscreteHistogram;
//Transformation
typedef M4D::Imaging::ProbabilityGrid		GridType;

typedef boost::filesystem::path			Path;
//typedef std::pair< Path, Path >			TrainingDataInfo;

struct TrainingDataInfo
{
	Path first;
	Path second;
	Path dir;
};

typedef std::vector< TrainingDataInfo >		TrainingDataInfos;

struct Transformation
{
	Vector< float32, 3 >
	operator()( const Vector< float32, 3 > &pos )const
	{
		Vector< float32, 3 > result = pos - _origin;
		result[2] *= _zScale;
		result += result[2] * _diff;
		return result;
	}

	Vector< float32, 3 >
	GetInversion( const Vector< float32, 3 > &pos )const
	{
		Vector< float32, 3 > result = pos;
		result -= result[2] * _diff;
		result[2] /= _zScale;

		return result + _origin;
	}

	float32 _zScale;
	Vector< float32, 3 > _origin;
	Vector< float32, 3 > _diff;
};

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

void
Train( const TrainingDataInfos &infos, Vector< uint32, 3 > size, Vector< float32, 3 > step, Vector< float32, 3 > origin, int32 minHist, int32 maxHist );

void
ConsolidateGeneralGrid( GridType &grid, int32 count );

void
GetTrainingSetInfos( const Path & dirPath, std::string indexExtension, TrainingDataInfos &infos, bool recursive = false );


#endif /*IMAGE_TOOLS_H*/
