#include "ImageTools.h"
#include <cmath>

#ifdef _MSC_VER 
	#include <cfloat>
	#define isnan _isnan
#endif

using namespace M4D::Imaging;
using namespace boost::filesystem;

static const float32 MaxLogarithm = 1.0e4f;

float32
RatioLogarithm( float32 a, float32 b )
{
	float32 tmp = log( a / b );
	if( isnan( tmp ) )
		return 0.0f;

	if( tmp < -MaxLogarithm )
		return -MaxLogarithm;

	if( tmp > MaxLogarithm )
		return MaxLogarithm;

	return tmp;
}

void
TrainingStep( 
		DiscreteHistogram	&generalInHistogram, 
		DiscreteHistogram	&generalOutHistogram,
		const ImageType		&image, 
		const MaskType		&mask,
		GridType		&generalGrid,
		int32 			minHist,
		int32			maxHist
	    )
{
	MaskType::PointType north;
	MaskType::PointType south;

	GetPoles( mask, north, south );

	Transformation tr = GetTransformation( north, south, mask.GetElementExtents() );

	DiscreteHistogram inHistogram( generalInHistogram.GetMin(), generalInHistogram.GetMax(), false );
	DiscreteHistogram outHistogram( generalOutHistogram.GetMin(), generalOutHistogram.GetMax(), true );

	/*D_PRINT( inHistogram );
	D_PRINT( outHistogram );*/

	D_PRINT( "Filling histograms..." );
	FillInOutHistograms( inHistogram, outHistogram, image, mask );

	generalInHistogram += inHistogram;
	generalOutHistogram += outHistogram;

	GridType grid( generalGrid.GetOrigin(), generalGrid.GetSize(), generalGrid.GetGridStep() );
	
	D_PRINT( "Filling grid..." );
	FillGrid( tr, image, mask, grid, minHist, maxHist );

	D_PRINT( "Incorporating grids..." );
	IncorporateGrids( grid, generalGrid );

}

GridPointRecord
ComputeProbRecord( Vector< float32, 3 > pos, Vector< float32, 3 > hStep, const ImageType &image, const MaskType &mask, int32 minHist, int32 maxHist )
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
	
	GridPointRecord result( 0.0, 0.0f, 0.0f, minHist, maxHist );

	Vector< int32, 3 > idx;
	int32 counter = 0;
	int32 inCounter = 0;
	for( idx[0] = rmin[0]; idx[0] <= rmax[0]; ++idx[0] ) {
		for( idx[1] = rmin[1]; idx[1] <= rmax[1]; ++idx[1] ) {
			for( idx[2] = rmin[2]; idx[2] <= rmax[2]; ++idx[2] ) {
				if( mask.GetElement( idx ) != 0 ) {
					++inCounter;
					result.inHistogram.IncCell( image.GetElement( idx ) );
				} else {
					result.outHistogram.IncCell( image.GetElement( idx ) );
				}	
				++counter;
			}
		}
	}
	if( counter == 0 ) {
		return GridPointRecord( 0.0f, 1.0f, 0.0f, minHist, maxHist );
	}
	float32 inProb = static_cast<float32>( inCounter ) / static_cast<float32>( counter );

	result.inProbabilityPos = inProb;
	result.outProbabilityPos = 1.0f - inProb;

	return result;
}

void
FillGrid( Transformation tr, const ImageType &image, const MaskType &mask, GridType &grid, int32 minHist, int32 maxHist )
{
	Vector< uint32, 3 > size = grid.GetSize();
	Vector< float32, 3 > step = grid.GetGridStep();
	Vector< float32, 3 > halfStep = 0.5f * step;
	Vector< float32, 3 > origin = grid.GetOrigin();
	Vector< float32, 3 > tmp;

	//TODO - halfstep correction
	//halfStep[2] *= (mask.GetMaximum()[2] - mask.GetMinimum()[2]) * mask.GetElementExtents()[2];

	Vector< uint32, 3 > idx;
	for( idx[0] = 0; idx[0] < size[0]; ++idx[0] ) {
		for( idx[1] = 0; idx[1] < size[1]; ++idx[1] ) {
			for( idx[2] = 0; idx[2] < size[2]; ++idx[2] ) {
				for( unsigned i = 0; i < 3; ++i ) {
					tmp[i] = idx[i]*step[i];
				}
				tmp -= origin;
				grid.GetPointRecord( idx ) = ComputeProbRecord( tr.GetInversion( tmp ), halfStep, image, mask, minHist, maxHist );
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
	MaskType::SliceRegion::PointType min = region.GetMinimum();
	MaskType::SliceRegion::PointType idx;
	MaskType::SliceRegion::PointType max = region.GetMaximum();
	int32 count = 0;
	for( idx = min; idx[1] < max[1]; ++idx[1] ) {
		for( idx[0] = min[0]; idx[0] < max[0]; ++idx[0] ) {
			//LOG( idx << " -> " << (int16)region.GetElement( idx ) );
			if( region.GetElement( idx ) != 0 ) {
				++count;
				sum += idx;
			}
		}
	}
	
	if( count == 0 ) {
		_THROW_ M4D::ErrorHandling::ExceptionBase( "Center of gravity unable to find." );
	}
	return MaskType::SliceRegion::PointType( sum[0] / count, sum[1] / count );
}

void
GetPoles( const MaskType & mask, MaskType::PointType &north, MaskType::PointType &south )
{
	int32 southSliceCoord = mask.GetMinimum()[2];
	int32 northSliceCoord = mask.GetMaximum()[2]-1;
	MaskType::SliceRegion southRegion = mask.GetSlice( southSliceCoord );
	MaskType::SliceRegion northRegion = mask.GetSlice( northSliceCoord );

	/*M4D::Imaging::Mask2D::Ptr tmp = mask.GetRestrictedImage( southRegion );
	ImageFactory::DumpImage( "pom.dump", *tmp );
	tmp = mask.GetRestrictedImage( northRegion );
	ImageFactory::DumpImage( "pom2.dump", *tmp );*/

	MaskType::SliceRegion::PointType southTmp = FindMaskCenterOfGravity( southRegion );
	MaskType::SliceRegion::PointType northTmp = FindMaskCenterOfGravity( northRegion );

	south = MaskType::PointType( southTmp[0], southTmp[1], southSliceCoord );
	north = MaskType::PointType( northTmp[0], northTmp[1], northSliceCoord );

	D_PRINT( "South pole = " << south );
	D_PRINT( "North pole = " << north );
}

void
IncorporateGrids( const GridType &last, GridType &general )
{
	Vector< uint32, 3 > size = last.GetSize();
	Vector< uint32, 3 > idx;
	for( idx[0] = 0; idx[0] < size[0]; ++idx[0] ) {
		for( idx[1] = 0; idx[1] < size[1]; ++idx[1] ) {
			for( idx[2] = 0; idx[2] < size[2]; ++idx[2] ) {
				const GridPointRecord &rec = last.GetPointRecord( idx );
				GridPointRecord &genRec = general.GetPointRecord( idx );
				genRec.inProbabilityPos += rec.inProbabilityPos;
				genRec.outProbabilityPos += rec.outProbabilityPos;

				genRec.inHistogram += rec.inHistogram;
				genRec.outHistogram += rec.outHistogram;
			}
		}
	}
}

void
ConsolidateGeneralGridRecord( GridPointRecord &rec, int32 count )
{
	rec.inProbabilityPos /= count;
	rec.outProbabilityPos /= count;

	//TODO - check extremes
	if( rec.inProbabilityPos < Epsilon ) {
		rec.inProbabilityPos = Epsilon;
	}
	if( rec.outProbabilityPos < Epsilon ) {
		rec.outProbabilityPos = Epsilon;
	}
	rec.logRatioPos = RatioLogarithm( rec.inProbabilityPos, rec.outProbabilityPos );

	double histogramSum = 0;
	int32 min = rec.inHistogram.GetMin();
	int32 max = rec.inHistogram.GetMax();

	for( int32 i = min; i < max; ++i ) {
		histogramSum += rec.inHistogram[i];
		histogramSum += rec.outHistogram[i];
	}
	for( int32 i = min; i < max; ++i ) {
		rec.inHistogram.SetValueCell( i, rec.inHistogram[i] / histogramSum );
		rec.outHistogram.SetValueCell( i, rec.outHistogram[i] / histogramSum );
		//rec.logHistogram.SetValueCell( i, RatioLogarithm( rec.inHistogram[i], rec.outHistogram[i] ) );
	}
	rec.inHistogram = HistogramPyramidSmooth( rec.inHistogram, 3 );
	rec.outHistogram = HistogramPyramidSmooth( rec.outHistogram, 3 );
	for( int32 i = min; i < max; ++i ) {
		rec.logHistogram.SetValueCell( i, RatioLogarithm( rec.inHistogram[i], rec.outHistogram[i] ) );
	}
}

void
ConsolidateGeneralGrid( GridType &grid, int32 count )
{
	Vector< uint32, 3 > size = grid.GetSize();
	Vector< uint32, 3 > idx;
	for( idx[0] = 0; idx[0] < size[0]; ++idx[0] ) {
		for( idx[1] = 0; idx[1] < size[1]; ++idx[1] ) {
			for( idx[2] = 0; idx[2] < size[2]; ++idx[2] ) {

				if( idx == Vector< uint32, 3 >( 20, 20, 10 ) ) {
					LOUT << grid.GetPointRecord( idx ).inHistogram << "\n\n\n---------------------------------------------------------------\n";
				}
				ConsolidateGeneralGridRecord( grid.GetPointRecord( idx ), count );
				if( idx == Vector< uint32, 3 >( 20, 20, 10 ) ) {
					LOUT << grid.GetPointRecord( idx ).inHistogram << "\n";
				}
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
			TrainingDataInfo info = RetrieveInfoFromIndex( itr->path() );
			info.dir = dirPath;
			infos.push_back( info );
		}
	}


}

void
NormalizeHistograms( 
		DiscreteHistogram	&generalInHistogram, 
		DiscreteHistogram	&generalOutHistogram, 
		FloatHistogram		&inHistogram, 
		FloatHistogram		&outHistogram, 
		FloatHistogram		&logRatioHistogram 
		)
{
	int32 min = inHistogram.GetMin();
	int32 max = inHistogram.GetMax();
	for( int32 i = min; i < max; ++i ) {
		inHistogram.SetValueCell( i, static_cast<float32>(generalInHistogram.Get( i )) / static_cast<float32>(generalInHistogram.GetSum()) );
		outHistogram.SetValueCell( i, static_cast<float32>(generalOutHistogram.Get( i )) / static_cast<float32>(generalOutHistogram.GetSum()) );

		logRatioHistogram.SetValueCell( i, RatioLogarithm( inHistogram.Get( i ), outHistogram.Get( i ) )  );
	}
	inHistogram.SetValueCell( min-1, 0.0f );
	inHistogram.SetValueCell( max, 0.0f );

	outHistogram.SetValueCell( min-1, 1.0f );
	outHistogram.SetValueCell( max, 1.0f );

	logRatioHistogram.SetValueCell( min-1, 2 * logRatioHistogram.Get( min )  );
	logRatioHistogram.SetValueCell( max, 2 * logRatioHistogram.Get( max-1 )  );
}

CanonicalProbModel::Ptr
Train( const TrainingDataInfos &infos, Vector< uint32, 3 > size, Vector< float32, 3 > step, Vector< float32, 3 > origin, int32 minHist, int32 maxHist )
{
	DiscreteHistogram generalInHistogram( minHist, maxHist, false );
	DiscreteHistogram generalOutHistogram( minHist, maxHist, true );

	GridType *generalGrid = new GridType( origin, size, step );

	//Set grid records two right values
	Vector< uint32, 3 > idx;
	GridPointRecord tmpRec( 0.0, 0.0f, 0.0f, minHist, maxHist );
	for( idx[0] = 0; idx[0] < size[0]; ++idx[0] ) {
		for( idx[1] = 0; idx[1] < size[1]; ++idx[1] ) {
			for( idx[2] = 0; idx[2] < size[2]; ++idx[2] ) {

				generalGrid->GetPointRecord( idx ) = tmpRec;
			}
		}
	}


	unsigned counter = 0;
	for( unsigned i = 0; i < infos.size(); ++i ) {
		ImageType::Ptr image;
		MaskType::Ptr mask;
		Path imageFile = infos[i].dir;
		imageFile /= infos[i].first;
		Path maskFile = infos[i].dir;
		maskFile /= infos[i].second;
		try {
			D_PRINT( "Loading training image number '" << i << "' from file '" << imageFile.string() <<"'." );
			AbstractImage::Ptr aimage = ImageFactory::LoadDumpedImage( imageFile.string() );
			image = ImageType::CastAbstractImage( aimage );

			D_PRINT( "Loading training mask number '" << i << "' from file '" << maskFile.string() <<"'." );
			aimage = ImageFactory::LoadDumpedImage( maskFile.string() );
			mask = MaskType::CastAbstractImage( aimage );
		} catch ( const M4D::ErrorHandling::ExceptionBase & e ) {
			continue;
		}

		++counter;
		TrainingStep( generalInHistogram, generalOutHistogram, *image, *mask, *generalGrid, minHist, maxHist );
	}	
	D_PRINT( "Consolidate general grid ..." );
	ConsolidateGeneralGrid( *generalGrid, counter );

	generalGrid->ComputeLayerStats();

	FloatHistogram *inHistogram = new FloatHistogram( minHist, maxHist, false );
	FloatHistogram *outHistogram = new FloatHistogram( minHist, maxHist, true );
	FloatHistogram *logRatioHistogram = new FloatHistogram( minHist, maxHist, true );

	D_PRINT( "Normalize histograms ..." );
	NormalizeHistograms( generalInHistogram, generalOutHistogram, *inHistogram, *outHistogram, *logRatioHistogram );

	CanonicalProbModel *model = new CanonicalProbModel( generalGrid, inHistogram, outHistogram, logRatioHistogram );


	LOG( "IN HISTOGRAM" );
	LOG( *inHistogram );
	LOG( "-----------------------------------------------" );
	LOG( "OUT HISTOGRAM" );
	LOG( *outHistogram );
	LOG( "-----------------------------------------------" );
	LOG( "LOG HISTOGRAM" );
	LOG( *logRatioHistogram );


/*	ImageType::Ptr tmp;
	tmp = MakeImageFromProbabilityGrid<InProbabilityAccessor>( *generalGrid, InProbabilityAccessor() );
	ImageFactory::DumpImage( "pom.dump", *tmp );

	tmp = MakeImageFromProbabilityGrid<OutProbabilityAccessor>( *generalGrid, OutProbabilityAccessor() );
	ImageFactory::DumpImage( "pom2.dump", *tmp );*/


	return CanonicalProbModel::Ptr( model );
}
