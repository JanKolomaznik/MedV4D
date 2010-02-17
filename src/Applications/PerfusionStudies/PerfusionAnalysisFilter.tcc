/**
 * @author Attila Ulman
 * @file PerfusionAnalysisFilter.tcc 
 * @{ 
 **/

#ifndef PERFUSION_ANALYSIS_FILTER_H
#error File PerfusionAnalysisFilter.tcc cannot be included directly!
#else


namespace M4D {
namespace Imaging {

template< typename ImageType >
PerfusionAnalysisFilter< ImageType >::PerfusionAnalysisFilter () 
  : PredecessorType( new Properties() )
{
  originalIntensities = smoothedIntensities = NULL;
  
	this->_inputPorts.AppendPort( new ImageInPort() );

  for ( uint8 i = 0; i < 3; i++ ) {
    this->_outputPorts.AppendPort( new ImageOutPort() );
  }

	this->_name = "PerfusionAnalysis";
}


template< typename ImageType >
PerfusionAnalysisFilter< ImageType >::PerfusionAnalysisFilter ( typename PerfusionAnalysisFilter< ImageType >::Properties *prop ) 
  : PredecessorType( prop )
{
  originalIntensities = smoothedIntensities = NULL;

  this->_inputPorts.AppendPort( new ImageInPort() );

  for ( uint8 i = 0; i < 3; i++ ) {
    this->_outputPorts.AppendPort( new ImageOutPort() );
  }

	this->_name = "PerfusionAnalysis";
}


template< typename ImageType >
PerfusionAnalysisFilter< ImageType >::~PerfusionAnalysisFilter () 
{
  delete [] originalIntensities;
  delete [] smoothedIntensities;
}


template< typename ImageType >
void PerfusionAnalysisFilter< ImageType >::Visualize ()
{
  VisOperator< ElementType > *visOperator = NULL;

	switch ( GetVisualizationType() ) 
  {
	  case VT_MAX:
		  VisualizeHelper( new MaxVisOperator< ElementType >( originalIntensities ) );
      break;

	  case VT_SUBTR:
		  VisualizeHelper( new SubtrVisOperator< ElementType >( originalIntensities,
                                                            GetSubtrIndexLow(), GetSubtrIndexHigh(), 
                                                            GetMaxCASliceIndex() ) );
      break;

    case VT_PARAM:
      visOperator = new ParamVisOperator< ElementType >( smoothedIntensities, GetParameterType(),
                                                         GetMinValuePercentage(), GetMaxValuePercentage() );
		  
      VisualizeHelper( visOperator );

      // setting the max/min values - for the viewer to calibrate the color ramp
      SetMaxParameterValue( static_cast< ParamVisOperator< ElementType > * >( visOperator )->maxValue );
      SetMinParameterValue( static_cast< ParamVisOperator< ElementType > * >( visOperator )->minValue );
      
      delete visOperator;
      break;

	  default:
		  ASSERT(false);
	}
}


template< typename ImageType >
void PerfusionAnalysisFilter< ImageType >::VisualizeHelper ( VisOperator< ElementType > *visOperator )
{
	IteratorsType outIt;

  for ( uint8 i = 0; i < 3; i++ ) {
    outIt.push_back( ((ImageType *)this->out[0])->GetIterator() );
  }

	typename ImageType::Iterator outEnd = outIt[0].End();

	uint32 idx = 0;
  
  while ( outIt[0] != outEnd ) 
  {
    visOperator->Result( idx, outIt );
		
    for ( uint8 i = 0; i < 3; i++ ) {
      ++outIt[i];
    }

    ++idx;
	}
}


template< typename ImageType >
bool PerfusionAnalysisFilter< ImageType >::ExecutionThreadMethod ( APipeFilter::UPDATE_TYPE utype )
{
  utype = utype;

	if ( readerBBox->WaitWhileDirty() != MS_MODIFIED ) 
	{
		for ( uint8 i = 0; i < 3; i++ ) {
      writerBBox[i]->SetState( MS_CANCELED );
    }

		return false;
	}

	bool result = this->Process();

	if ( result ) 
  {
    for ( uint8 i = 0; i < 3; i++ ) {
		  writerBBox[i]->SetModified();
    }
	} 
  else 
  {
    for ( uint8 i = 0; i < 3; i++ ) {
		  writerBBox[i]->SetState( MS_CANCELED );
    }
	}
	
  return result;
}


template< typename ImageType >
void PerfusionAnalysisFilter< ImageType >::PrepareOutputDatasets ()
{
  PredecessorType::PrepareOutputDatasets();

	int32 minimums[ ImageTraits< ImageType >::Dimension ];
	int32 maximums[ ImageTraits< ImageType >::Dimension ];
	
  float32 voxelExtents[ ImageTraits< ImageType >::Dimension ];

	for ( unsigned i = 0; i < ImageTraits< ImageType >::Dimension; ++i ) 
  {
		const DimensionExtents &dimExt = this->in[0]->GetDimensionExtents( i );

		minimums[i] = dimExt.minimum;
		maximums[i] = dimExt.maximum;
		
    voxelExtents[i] = dimExt.elementExtent;
	}

  maximums[ImageTraits< ImageType >::Dimension - 1] = GetExaminedSliceNum();

  for ( uint8 i = 0; i < 3; i++ ) {
    this->SetOutputImageSize( i, ImageTraits< ImageType >::Dimension, minimums, maximums, voxelExtents ); 
  }
}


template< typename ImageType >
void PerfusionAnalysisFilter< ImageType >::BeforeComputation ( APipeFilter::UPDATE_TYPE &utype )
{
	PredecessorType::BeforeComputation( utype );
}


template< typename ImageType >
void PerfusionAnalysisFilter< ImageType >::MarkChanges ( APipeFilter::UPDATE_TYPE utype )
{
  utype = utype;
	
  readerBBox = this->in[0]->GetWholeDirtyBBox();
	for ( uint8 i = 0; i < 3; i++ ) {
    writerBBox[i] = &(this->out[i]->SetWholeDirtyBBox());
  }
}


template< typename ImageType >
void PerfusionAnalysisFilter< ImageType >::AfterComputation ( bool successful )
{
  PredecessorType::AfterComputation( successful );
}


template< typename ImageType >
bool PerfusionAnalysisFilter< ImageType >::Process ()
{
  const ImageType *inImage = (const ImageType *)this->in[0]; 
  
  Vector< uint32, 3 > size;
	Vector< int32, 3 > strides;
	ElementType *pointer = inImage->GetPointer( size, strides );

  int32 xStride = strides[0];
  int32 yStride = strides[1];
  int32 zStride = strides[2];

  uint32 width  = size[0];
  uint32 height = size[1];

	uint32 depth    = size[2];
  uint32 sliceNum = GetExaminedSliceNum();
  uint32 times    = (uint32)(depth / sliceNum);

  uint32 sliceStride = height * width; 
  originalIntensities = new IntensitiesType[ sliceNum * sliceStride ];

  SumType sliceSums;

  for ( uint32 slice = 0; slice < sliceNum; ++slice )
  {
    // 1 point (x, y) in different times (t)
    for ( uint32 t = 0; t < times; ++t ) 
    {
      uint32 idx = slice * sliceStride;

      int32 sum = 0;

      // loop over 1 timesequence (1 slice in different times)
	    for ( uint32 y = 0; y < height; ++y ) 
      {
        ElementType *in = pointer + y * yStride;

		    for ( uint32 x = 0; x < width; ++x, idx++ ) 
        {
			    originalIntensities[idx].push_back( *in );

          sum += *in;

			    in += xStride;
		    }
      }

      sliceSums.push_back( sum );  

      pointer += zStride;
    }
  }

  // setting the size of the intensities vector - for the settingsbox configuration
  SetIntensitiesSize( times );

  // setting the index of slice with maximum CA (summed intensities) - for the settingsbox configuration 
  // - prefill the spinbox with estimation of the optimal index
  uint16 resultIndex = 0; 

  for ( uint32 i = 0; i < sliceNum; ++i ) 
  {
    SumType::iterator begin = sliceSums.begin() + i * times;
    SumType::iterator end   = begin + times;
    
    resultIndex += max_element( begin, end ) - begin;
  }

  SetMaxCASliceIndex( resultIndex / sliceNum );

  // smoothing TICs
  smoothedIntensities = new IntensitiesType[ sliceNum * sliceStride ];

  float kernel[] = { 0.05, 0.09, 0.16, 0.4, 0.16, 0.09, 0.05 };    // simple Gaussian-like kernel

  for ( uint32 idx = 0; idx < sliceNum * sliceStride; idx++ ) 
  {
    // start convolution from intensities[idx][KERNEL_SIZE - 1] to intensities[idx][size - 1] (last)
    for ( uint16 i = KERNEL_SIZE - 1; i < originalIntensities[idx].size(); ++i )
    {
      ElementType out = 0;                                         // init to 0 before accumulate

      for ( int16 j = i, k = 0; k < KERNEL_SIZE; --j, ++k ) {
        out += originalIntensities[idx][j] * kernel[k];
      }

      smoothedIntensities[idx].push_back( out );
    } 
  }

  // actual visualization
  Visualize();

  return true;
}

} // namespace Imaging
} // namespace M4D


#endif // PERFUSION_ANALYSIS_FILTER_H

/** @} */

