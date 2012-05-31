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
	GraphCutSegmentationWrapper(): mForegroundMarkers( boost::make_shared< std::set<uint32> >() ), mBackgroundMarkers( boost::make_shared< std::set<uint32> >() )
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
			TTYPE threshold = TypeTraits<TTYPE>::Max;
			
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
			
			LocalMinimaRegions3D( gradientImage->GetRegion(), mWatersheds->GetRegion(), threshold );

			WatershedTransformation3D( mWatersheds->GetRegion(), gradientImage->GetRegion(), mWatersheds->GetRegion() );
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
	}
	
	M4D::Imaging::AImage::Ptr mInputImage;
	
	M4D::Imaging::Image< uint32, 3 >::Ptr mWatersheds;
	M4D::Imaging::AImage::Ptr mGradientImage;
	
	WeightedEdgeListGraph::Ptr mGraph;
	
	boost::shared_ptr< std::set< uint32 > > mForegroundMarkers;
	boost::shared_ptr< std::set< uint32 > > mBackgroundMarkers;
};


#endif //GRAPH_CUT_SEGMENTATION