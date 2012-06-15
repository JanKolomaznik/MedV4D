#ifndef GRAPH_CUT_SEGMENTATION
#define GRAPH_CUT_SEGMENTATION


#include "MedV4D/Imaging/Image.h"

#include "MedV4D/Imaging/cuda/ConnectedComponentLabeling.h"
#include "MedV4D/Imaging/cuda/EdgeDetection.h"
#include "MedV4D/Imaging/cuda/LocalMinimaDetection.h"
#include "MedV4D/Imaging/cuda/WatershedTransformation.h"
#include "MedV4D/Imaging/cuda/GraphOperations.h"
#include "MedV4D/Imaging/cuda/SimpleFilters.h"

#include <boost/shared_ptr.hpp>


class GraphCutSegmentationWrapper
{
public:
	typedef std::set<uint32> MarkerSet;
	
	GraphCutSegmentationWrapper(): mForegroundMarkers( boost::make_shared< MarkerSet >() ), mBackgroundMarkers( boost::make_shared< MarkerSet >() )
	{ }
	
	void
	setInputImage( M4D::Imaging::AImage::Ptr aInputImage )
	{
		mInputImage = aInputImage;
	}
	
	void
	reset()
	{
		mInputImage.reset();
		mWatersheds.reset();
		mGradientImage.reset();
		mForegroundMarkers = boost::make_shared< std::set<uint32> >();
		mBackgroundMarkers = boost::make_shared< std::set<uint32> >();
	}
//protected:
	void
	computeGradientImage()
	{
		ASSERT( mInputImage );
		
		NUMERIC_TYPE_TEMPLATE_SWITCH_MACRO( mInputImage->GetElementTypeID(),
			//TTYPE threshold = TypeTraits<TTYPE>::Max;
			
			typedef M4D::Imaging::Image< TTYPE, 3 > IMAGE_TYPE;
			IMAGE_TYPE::Ptr typedImage = IMAGE_TYPE::Cast( mInputImage );
			IMAGE_TYPE::Ptr gradientImage = M4D::Imaging::ImageFactory::CreateEmptyImageFromExtents< TTYPE, 3 >( typedImage->GetMinimum(), typedImage->GetMaximum(), typedImage->GetElementExtents() );

			Sobel3D( typedImage->GetRegion(), gradientImage->GetRegion(), static_cast< TTYPE >( 0 ) );
			mGradientImage = gradientImage;
			LOG( "Computed image gradient" );
		);
	}
	
	void
	computeWatershedTransformation()
	{
		ASSERT( mInputImage );
		ASSERT( mGradientImage );
		NUMERIC_TYPE_TEMPLATE_SWITCH_MACRO( mGradientImage->GetElementTypeID(),
			TTYPE threshold = TypeTraits<TTYPE>::Max;
			
			typedef M4D::Imaging::Image< TTYPE, 3 > IMAGE_TYPE;
			IMAGE_TYPE::Ptr gradientImage = IMAGE_TYPE::Cast( mGradientImage );		
			
			mWatersheds = M4D::Imaging::ImageFactory::CreateEmptyImageFromExtents< uint32, 3 >( gradientImage->GetMinimum(), gradientImage->GetMaximum(), gradientImage->GetElementExtents() );
			
			mRegionCount = localMinimaRegions3D( gradientImage->GetRegion(), mWatersheds->GetRegion(), threshold );

			watershedTransformation3D( mWatersheds->GetRegion(), gradientImage->GetRegion(), mWatersheds->GetRegion() );
			
			LOG( "Computed watershed transformation" );
		);
	}
	
	void
	buildNeighborhoodGraph()
	{
		mGraph = boost::make_shared<WeightedEdgeListGraph>();
		
		NUMERIC_TYPE_TEMPLATE_SWITCH_MACRO( mGradientImage->GetElementTypeID(),
			typedef M4D::Imaging::Image< TTYPE, 3 > IMAGE_TYPE;
			IMAGE_TYPE::Ptr gradientImage = IMAGE_TYPE::Cast( mGradientImage );
			createAdjacencyGraph( *mGraph, mWatersheds->GetRegion(), gradientImage->GetRegion() );
		);
		
		LOG( boost::str( boost::format( "Graph info :\nVertex count = %1% \nEdge count = %2%" ) % mGraph->mVertexCount % mGraph->mEdgeCount ) );
	}
	
	void
	extendGraph()
	{
		ASSERT( mGraph );
		ASSERT( mForegroundMarkers );
		ASSERT( mBackgroundMarkers );
		
		mExtendedGraph = boost::make_shared<WeightedEdgeListGraph>();
		*mExtendedGraph = *mGraph;
		
		mSourceID = mGraph->mVertexCount + 1;
		mSinkID = mGraph->mVertexCount + 2;
		mExtendedGraph->mVertexCount += 2;
		mExtendedGraph->mEdges.reserve( mGraph->mEdgeCount + mForegroundMarkers->size() + mBackgroundMarkers->size() + 10 );
		
		for( MarkerSet::iterator it = mForegroundMarkers->begin(); it != mForegroundMarkers->end(); ++it ) {
			mExtendedGraph->mEdges.push_back( WeightedEdgeListGraph::EdgeRecord( mSourceID, *it ) );
			mExtendedGraph->mWeights.push_back( 1000 );
			++mExtendedGraph->mEdgeCount;
		}
		for( MarkerSet::iterator it = mBackgroundMarkers->begin(); it != mBackgroundMarkers->end(); ++it ) {
			mExtendedGraph->mEdges.push_back( WeightedEdgeListGraph::EdgeRecord( mSinkID, *it ) );
			mExtendedGraph->mWeights.push_back( 1000 );
			++mExtendedGraph->mEdgeCount;
		}
		ASSERT( mExtendedGraph->mEdgeCount == mExtendedGraph->mEdges.size() );
		ASSERT( mExtendedGraph->mEdgeCount == mExtendedGraph->mWeights.size() );
		LOG( boost::str( boost::format( "Adding %1% foreground marker edges and %2% background marker edges" ) % mForegroundMarkers->size() % mBackgroundMarkers->size() ) );
	}

	void
	executeGraphCut()
	{
		ASSERT( mExtendedGraph );

		std::vector< bool > componentSet;

		minGraphCut( *mExtendedGraph, componentSet, mSourceID, mSinkID );

	}
	
	M4D::Imaging::AImage::Ptr mInputImage;
	
	M4D::Imaging::Image< uint32, 3 >::Ptr mWatersheds;
	size_t mRegionCount;
	M4D::Imaging::AImage::Ptr mGradientImage;
	
	WeightedEdgeListGraph::Ptr mGraph;
	
	WeightedEdgeListGraph::Ptr mExtendedGraph;
	size_t mSourceID;
	size_t mSinkID;
	
	boost::shared_ptr< MarkerSet > mForegroundMarkers;
	boost::shared_ptr< MarkerSet > mBackgroundMarkers;
};


#endif //GRAPH_CUT_SEGMENTATION
