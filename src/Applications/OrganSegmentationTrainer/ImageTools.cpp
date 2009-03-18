#include "ImageTools.h"


using namespace M4D::Imaging;
using namespace boost::filesystem;

Transformation
GetTransformation( MaskType::PointType north, MaskType::PointType south, Vector< float32, 3 > elExtents )
{
	M4D::ErrorHandling::ETODO();
	return Transformation();
}

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

	Transformation tr = GetTransformation( north, south, mask.GetElementExtents() );

	DiscreteHistogram inHistogram( generalInHistogram.GetMin(), generalInHistogram.GetMax(), false );
	DiscreteHistogram outHistogram( generalOutHistogram.GetMin(), generalOutHistogram.GetMax(), true );

	FillInOutHistograms( inHistogram, outHistogram, image, mask );

	generalInHistogram += inHistogram;
	generalOutHistogram += outHistogram;

	GridType grid( generalGrid.GetOrigin(), generalGrid.GetSize(), generalGrid.GetGridStep() );
	
	FillGrid( tr.GetInversion(), image, mask, grid );

	IncorporateGrids( grid, generalGrid );

}

GridPointRecord
ComputeProbRecord( Vector< float32, 3 > pos, Vector< float32, 3 > hStep, const ImageType &image, const MaskType &mask )
{
	Vector< float32, 3 > min = pos - hStep;
	Vector< float32, 3 > max = pos + hStep;
	Vector< float32, 3 > extents = mask.GetElementExtents();

	Vector< int32, 3 > rmin = mask.GetMinimum();
	Vector< int32, 3 > rmax = mask.GetMaximum();

	for( unsigned i = 0; i < 3; ++i ) {
		rmin[i] = Max( ROUND( min[i] / extents[i] ), rmin[i] );
		rmax[i] = Min( ROUND( max[i] / extents[i] ), rmax[i]-1 ) ;
	}
	
	Vector< int32, 3 > idx;
	int32 counter = 0;
	int32 inCounter = 0;
	for( idx[0] = rmin[0]; idx[0] <= max[0]; ++idx[0] ) {
		for( idx[1] = rmin[1]; idx[1] <= max[1]; ++idx[1] ) {
			for( idx[2] = rmin[2]; idx[2] <= max[0]; ++idx[2] ) {
				if( mask.GetElement( idx ) != 0 ) {
					++inCounter;
				}	
				++counter;
			}
		}
	}
	if( counter == 0 ) {
		return GridPointRecord( 0.0f, 1.0f, 0.0f );
	}
	float32 inProb = static_cast<float32>( inCounter ) / static_cast<float32>( counter );

	return GridPointRecord( inProb, 1.0f - inProb, 0.0f );
}

void
FillGrid( Transformation invTr, const ImageType &image, const MaskType &mask, GridType &grid )
{
	Vector< uint32, 3 > size = grid.GetSize();
	Vector< float32, 3 > step = grid.GetGridStep();
	Vector< float32, 3 > halfStep = 0.5f * step;
	Vector< float32, 3 > origin = grid.GetOrigin();
	Vector< float32, 3 > tmp;

	//TODO - halfstep correction
	halfStep[2] *= (mask.GetMaximum()[2] - mask.GetMinimum()[2]) * mask.GetElementExtents()[2];

	Vector< uint32, 3 > idx;
	for( idx[0] = 0; idx[0] < size[0]; ++idx[0] ) {
		for( idx[1] = 0; idx[1] < size[1]; ++idx[1] ) {
			for( idx[2] = 0; idx[2] < size[0]; ++idx[2] ) {
				for( unsigned i = 0; i < 3; ++i ) {
					tmp[i] = idx[i]*step[i];
				}
				tmp += origin;
				grid.GetPointRecord( idx ) = ComputeProbRecord( invTr( tmp ), halfStep, image, mask );
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

void
IncorporateGrids( const GridType &last, GridType &general )
{
	Vector< uint32, 3 > size = last.GetSize();
	Vector< uint32, 3 > idx;
	for( idx[0] = 0; idx[0] < size[0]; ++idx[0] ) {
		for( idx[1] = 0; idx[1] < size[1]; ++idx[1] ) {
			for( idx[2] = 0; idx[2] < size[0]; ++idx[2] ) {
				const GridPointRecord &rec = last.GetPointRecord( idx );
				GridPointRecord &genRec = general.GetPointRecord( idx );
				genRec.inProbabilityPos += rec.inProbabilityPos;
				genRec.outProbabilityPos += rec.outProbabilityPos;
			}
		}
	}
}

void
ConsolidateGeneralGrid( GridType &grid, int32 count )
{
	Vector< uint32, 3 > size = grid.GetSize();
	Vector< uint32, 3 > idx;
	for( idx[0] = 0; idx[0] < size[0]; ++idx[0] ) {
		for( idx[1] = 0; idx[1] < size[1]; ++idx[1] ) {
			for( idx[2] = 0; idx[2] < size[0]; ++idx[2] ) {
				GridPointRecord &rec = grid.GetPointRecord( idx );
				rec.inProbabilityPos /= count;
				rec.outProbabilityPos /= count;

				//TODO - check extremes
				rec.logRationPos = log( rec.inProbabilityPos / rec.outProbabilityPos );
			}
		}
	}
	
}

TrainingDataInfo
RetrieveInfoFromIndex( const Path & indexPath )
{
	std::fstream file( indexPath.string().data() );

	TrainingDataInfo info;

	file >> info.first;
	file >> info.second;

	return info;
}

void
GetTrainingSetInfos( const Path & dirPath, std::string indexExtension, TrainingDataInfos &infos, bool recursive )
{
	//pairs.clear();
	if ( !exists( dirPath ) ) return;

	directory_iterator end_itr; // default construction yields past-the-end
	for ( directory_iterator itr( dirPath ); itr != end_itr; ++itr )
	{
		if ( is_directory(itr->status()) & recursive )
		{
			GetTrainingSetInfos( itr->path(), indexExtension, infos, recursive );
		} else if ( itr->path().extension() == indexExtension ) {
			D_PRINT( "Found index file : " << itr->path() );
			infos.push_back( RetrieveInfoFromIndex( itr->path() ) );
		}
	}


}

void
Train( const TrainingDataInfos &infos, Vector< uint32, 3 > size, Vector< float32, 3 > step, Vector< float32, 3 > origin, int32 minHist, int32 maxHist )
{
	DiscreteHistogram generalInHistogram( minHist, maxHist, false );
	DiscreteHistogram generalOutHistogram( minHist, maxHist, true );

	GridType generalGrid( origin, size, step );

	unsigned counter = 0;
	for( unsigned i = 0; i < infos.size(); ++i ) {
		ImageType::Ptr image;
		MaskType::Ptr mask;
		try {
			AbstractImage::Ptr aimage = ImageFactory::LoadDumpedImage( infos[i].first.string() );
			image = ImageType::CastAbstractImage( aimage );
			aimage = ImageFactory::LoadDumpedImage( infos[i].second.string() );
			mask = MaskType::CastAbstractImage( aimage );
		} catch (...) {
			continue;
		}

		++counter;
		TrainingStep( generalInHistogram, generalOutHistogram, *image, *mask, generalGrid );
	}	
}
