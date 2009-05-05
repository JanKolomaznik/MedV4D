#ifndef CELLREADYTHRESHOLDSEGMENTATIONLEVELSETIMAGEFILTER_H_
#error File initPartOfFilter.tcc cannot be included directly!
#else

#include "itkZeroCrossingImageFilter.h"
#include "itkImageRegionIterator.h"
#include "itkImageRegionConstIterator.h"
#include "itkShiftScaleImageFilter.h"
#include "itkNeighborhoodAlgorithm.h"

namespace itk {

///////////////////////////////////////////////////////////////////////////////


template<class TInputImage,class TFeatureImage, class TOutputPixelType>
MySegmtLevelSetFilter_InitPart<TInputImage, TFeatureImage, TOutputPixelType>
::MySegmtLevelSetFilter_InitPart()
	: _workManager(m_SPEManager.GetSPECount())
{		  
	m_SPEManager._workManager = &_workManager;
  m_IsoSurfaceValue = this->m_ValueZero;
  this->SetRMSChange(static_cast<double>(this->m_ValueZero));
  m_BoundsCheckingActive = false;
  m_ConstantGradientValue = 1.0;

  this->SetIsoSurfaceValue(NumericTraits<ValueType>::Zero);
  
  // Provide some reasonable defaults which will at least prevent infinite
  // looping.
  this->SetMaximumRMSError(0.02);
  this->SetNumberOfIterations(1000);
  
  //initial properties
  m_runConf.m_upThreshold = 500;
  m_runConf.m_downThreshold = -500;
  m_runConf.m_propWeight = 1;
  m_runConf.m_curvWeight = 0.001f;
  m_runConf.m_ConstantGradientValue = this->m_ConstantGradientValue;
}
///////////////////////////////////////////////////////////////////////////////
template<class TInputImage,class TFeatureImage, class TOutputPixelType>
MySegmtLevelSetFilter_InitPart<TInputImage, TFeatureImage, TOutputPixelType>
::~MySegmtLevelSetFilter_InitPart()
{
}
///////////////////////////////////////////////////////////////////////////////
template<class TInputImage,class TFeatureImage, class TOutputPixelType>
void
MySegmtLevelSetFilter_InitPart<TInputImage, TFeatureImage, TOutputPixelType>
::CopyInputToOutput()
{
  // This method is the first step in initializing the level-set image, which
  // is also the output of the filter.  The input is passed through a
  // zero crossing filter, which produces zero's at pixels closest to the zero
  // level set and one's elsewhere.  The actual zero level set values will be
  // adjusted in the Initialize() step to more accurately represent the
  // position of the zero level set.

  // First need to subtract the iso-surface value from the input image.
  typedef ShiftScaleImageFilter<TInputImage, OutputImageType> ShiftScaleFilterType;
  typename ShiftScaleFilterType::Pointer shiftScaleFilter = ShiftScaleFilterType::New();
  shiftScaleFilter->SetInput( this->GetInput()  );
  shiftScaleFilter->SetShift( - m_IsoSurfaceValue );
  // keep a handle to the shifted output
  m_ShiftedImage = shiftScaleFilter->GetOutput();
  
  typename ZeroCrossingImageFilter<OutputImageType, OutputImageType>::Pointer
    zeroCrossingFilter = ZeroCrossingImageFilter<OutputImageType,
    OutputImageType>::New();
  zeroCrossingFilter->SetInput(m_ShiftedImage);
  zeroCrossingFilter->GraftOutput(this->GetOutput());
  zeroCrossingFilter->SetBackgroundValue(this->m_ValueOne);
  zeroCrossingFilter->SetForegroundValue(this->m_ValueZero);

  zeroCrossingFilter->Update();

  this->GraftOutput(zeroCrossingFilter->GetOutput());
}

///////////////////////////////////////////////////////////////////////////////
template<class TInputImage,class TFeatureImage, class TOutputPixelType>
void
MySegmtLevelSetFilter_InitPart<TInputImage, TFeatureImage, TOutputPixelType>
::InitializeInputAndConstructLayers()
{
  unsigned int i;

  if (this->GetUseImageSpacing())
    {
    double minSpacing = NumericTraits<double>::max();
    for (i=0; i<OutputImageType::ImageDimension; i++)
      {
      minSpacing = vnl_math_min(minSpacing,this->GetInput()->GetSpacing()[i]);
      }
    m_ConstantGradientValue = minSpacing;
    }
  else
    {
    m_ConstantGradientValue = 1.0;
    }

  // Allocate the status image.
  m_StatusImage = StatusImageType::New();
  m_StatusImage->SetRegions(this->GetOutput()->GetRequestedRegion());
  m_StatusImage->Allocate();

  // Initialize the status image to contain all m_StatusNull values.
  ImageRegionIterator<StatusImageType>
    statusIt(m_StatusImage, m_StatusImage->GetRequestedRegion());
  for (statusIt.GoToBegin(); ! statusIt.IsAtEnd(); ++statusIt)
    {
    statusIt.Set( this->m_StatusNull );
    }

  // Initialize the boundary pixels in the status image to
  // m_StatusBoundaryPixel values.  Uses the face calculator to find all of the
  // region faces.
  typedef NeighborhoodAlgorithm::ImageBoundaryFacesCalculator<StatusImageType>
    BFCType;

  BFCType faceCalculator;
  typename BFCType::FaceListType faceList;
  typename BFCType::SizeType sz;
  typename BFCType::FaceListType::iterator fit;
  
  sz.Fill(1);
  faceList = faceCalculator(m_StatusImage, m_StatusImage->GetRequestedRegion(), sz);
  fit = faceList.begin();

  for (++fit; fit != faceList.end(); ++fit) // skip the first (nonboundary) region
    {
    statusIt = ImageRegionIterator<StatusImageType>(m_StatusImage, *fit);
    for (statusIt.GoToBegin(); ! statusIt.IsAtEnd(); ++statusIt)
      {
      statusIt.Set( this->m_StatusBoundaryPixel );
      }
    }

  
  // Construct the active layer and initialize the first layers inside and
  // outside of the active layer.
  this->ConstructActiveLayer();

  // Construct the rest of the non-active set layers using the first two
  // layers. Inside layers are odd numbers, outside layers are even numbers.
  for (i = 1; i < LYERCOUNT - 2; ++i)
    {
    this->ConstructLayer(i, i+2);
    }
}

///////////////////////////////////////////////////////////////////////////////
template<class TInputImage,class TFeatureImage, class TOutputPixelType>
void
MySegmtLevelSetFilter_InitPart<TInputImage, TFeatureImage, TOutputPixelType>
::InitRunConf()
{	
	// feature image
    m_runConf.featureImageProps.imageData = 
    	(FeaturePixelType *)GetFeatureImage()->GetBufferPointer();
    m_runConf.featureImageProps.region = 
    	ConvertRegion<TFeatureImage, M4D::Cell::TRegion>(*GetFeatureImage());
    m_runConf.featureImageProps.spacing = 
    	ConvertIncompatibleVectors<M4D::Cell::TSpacing, typename TFeatureImage::SpacingType>(GetFeatureImage()->GetSpacing());
    // output image
    m_runConf.valueImageProps.imageData = (ValueType *)this->GetOutput()->GetBufferPointer();
    m_runConf.valueImageProps.region = 
    	ConvertRegion<OutputImageType, M4D::Cell::TRegion>(*this->GetOutput());
    m_runConf.valueImageProps.spacing = ConvertIncompatibleVectors<M4D::Cell::TSpacing, typename OutputImageType::SpacingType>(this->GetOutput()->GetSpacing());
    //status image
    m_runConf.statusImageProps.imageData = (StatusType *)m_StatusImage->GetBufferPointer();
    m_runConf.statusImageProps.region = 
    	ConvertRegion<StatusImageType, M4D::Cell::TRegion>(*m_StatusImage);
    m_runConf.statusImageProps.spacing = ConvertIncompatibleVectors<M4D::Cell::TSpacing, typename StatusImageType::SpacingType>(m_StatusImage->GetSpacing());
    
    _workManager.SetupRunConfig(&m_runConf);
}

///////////////////////////////////////////////////////////////////////////////
template<class TInputImage,class TFeatureImage, class TOutputPixelType>
void
MySegmtLevelSetFilter_InitPart<TInputImage, TFeatureImage, TOutputPixelType>
::InitializeBackgroundPixels()
{
  // Assign background pixels OUTSIDE the sparse field layers to a new level set
  // with value greater than the outermost layer.  Assign background pixels
  // INSIDE the sparse field layers to a new level set with value less than
  // the innermost layer.
  const ValueType max_layer = static_cast<ValueType>(NUM_LAYERS);

  const ValueType outside_value  = (max_layer+1) * m_ConstantGradientValue;
  const ValueType inside_value = -(max_layer+1) * m_ConstantGradientValue;
 
  ImageRegionConstIterator<StatusImageType> statusIt(m_StatusImage,
                                                     this->GetOutput()->GetRequestedRegion());

  ImageRegionIterator<OutputImageType> outputIt(this->GetOutput(),
                                                this->GetOutput()->GetRequestedRegion());

  ImageRegionConstIterator<OutputImageType> shiftedIt(m_ShiftedImage,
                                                      this->GetOutput()->GetRequestedRegion());
  
  for (outputIt = outputIt.Begin(), statusIt = statusIt.Begin(),
         shiftedIt = shiftedIt.Begin();
       ! outputIt.IsAtEnd(); ++outputIt, ++statusIt, ++shiftedIt)
    {
    if (statusIt.Get() == this->m_StatusNull || statusIt.Get() == this->m_StatusBoundaryPixel)
      {
      if (shiftedIt.Get() > this->m_ValueZero)
        {
        outputIt.Set(outside_value);
        }
      else
        {
        outputIt.Set(inside_value);
        }
      }
    }
  
};
///////////////////////////////////////////////////////////////////////////////
template<class TInputImage,class TFeatureImage, class TOutputPixelType>
void
MySegmtLevelSetFilter_InitPart<TInputImage, TFeatureImage, TOutputPixelType>
::ConstructActiveLayer()
{
  //
  // We find the active layer by searching for 0's in the zero crossing image
  // (output image).  The first inside and outside layers are also constructed
  // by searching the neighbors of the active layer in the (shifted) input image.
  // Negative neighbors not in the active set are assigned to the inside,
  // positive neighbors are assigned to the outside.
  //
  // During construction we also check whether any of the layers of the active
  // set (or the active set itself) is sitting on a boundary pixel location. If
  // this is the case, then we need to do active bounds checking in the solver.
  //
  
  unsigned int i;
  NeighborhoodIterator<OutputImageType>
    shiftedIt(m_NeighborList.GetRadius(), m_ShiftedImage,
              this->GetOutput()->GetRequestedRegion());
  NeighborhoodIterator<OutputImageType>
    outputIt(m_NeighborList.GetRadius(), this->GetOutput(),
             this->GetOutput()->GetRequestedRegion());
  NeighborhoodIterator<StatusImageType>
    statusIt(m_NeighborList.GetRadius(), m_StatusImage,
             this->GetOutput()->GetRequestedRegion());
  IndexType center_index, offset_index;
  bool bounds_status;
  ValueType value;
  StatusType layer_number;

  typename OutputImageType::IndexType upperBounds, lowerBounds;
  lowerBounds = this->GetOutput()->GetRequestedRegion().GetIndex();
  upperBounds = this->GetOutput()->GetRequestedRegion().GetIndex()
    + this->GetOutput()->GetRequestedRegion().GetSize();

  for (outputIt.GoToBegin(); !outputIt.IsAtEnd(); ++outputIt)
    {
    if ( outputIt.GetCenterPixel() == this->m_ValueZero )
      {
      // Grab the neighborhood in the status image.
      center_index = outputIt.GetIndex();
      statusIt.SetLocation( center_index );

      // Check to see if any of the sparse field touches a boundary.  If so,
      // then activate bounds checking.
      for (i = 0; i < OutputImageType::ImageDimension; i++)
        {
        if (center_index[i] + static_cast<long>(NUM_LAYERS) >= (upperBounds[i] - 1)
            || center_index[i] - static_cast<long>(NUM_LAYERS) <= lowerBounds[i])
          {
          m_BoundsCheckingActive = true;
          }
        }
      
      _workManager.PUSHNode(center_index, 0);
      
      statusIt.SetCenterPixel( 0 );

      // Grab the neighborhood in the image of shifted input values.
      shiftedIt.SetLocation( center_index );
          
      // Search the neighborhood pixels for first inside & outside layer
      // members.  Construct these lists and set status list values. 
      for (i = 0; i < m_NeighborList.GetSize(); ++i)
        {
        offset_index = center_index
          + m_NeighborList.GetNeighborhoodOffset(i);

        if ( outputIt.GetPixel(m_NeighborList.GetArrayIndex(i)) != this->m_ValueZero)
          {
          value = shiftedIt.GetPixel(m_NeighborList.GetArrayIndex(i));

          if ( value < this->m_ValueZero ) // Assign to first inside layer.
            {
            layer_number = 1;
            }
          else // Assign to first outside layer
            {
            layer_number = 2;
            }
                  
          statusIt.SetPixel( m_NeighborList.GetArrayIndex(i),
                             layer_number, bounds_status );
          if ( bounds_status == true ) // In bounds.
            {
        	  _workManager.PUSHNode(offset_index, layer_number);
            } // else do nothing.
          }
        }
      }
    }
}
///////////////////////////////////////////////////////////////////////////////
template<class TInputImage,class TFeatureImage, class TOutputPixelType>
void
MySegmtLevelSetFilter_InitPart<TInputImage, TFeatureImage, TOutputPixelType>
::ConstructLayer(StatusType from, StatusType to)
{
  unsigned int i;
  bool boundary_status;
  typename LayerType::ConstIterator fromIt;
  NeighborhoodIterator<StatusImageType>
    statusIt(m_NeighborList.GetRadius(), m_StatusImage,
             this->GetOutput()->GetRequestedRegion() );
  
  for(uint32 spuIt=0; spuIt<m_SPEManager.GetSPECount(); spuIt++)
  {
  // For all indicies in the "from" layer...
  for (fromIt = _workManager.GetLayers()[spuIt].layers[from]->Begin();
       fromIt != _workManager.GetLayers()[spuIt].layers[from]->End();  ++fromIt)
    {
    // Search the neighborhood of this index in the status image for
    // unassigned indicies. Push those indicies onto the "to" layer and
    // assign them values in the status image.  Status pixels outside the
    // boundary will be ignored.
    statusIt.SetLocation( fromIt->m_Value );
    for (i = 0; i < m_NeighborList.GetSize(); ++i)
      {
      if ( statusIt.GetPixel( m_NeighborList.GetArrayIndex(i) )
           == this->m_StatusNull )
        {
        statusIt.SetPixel(m_NeighborList.GetArrayIndex(i), to,
                          boundary_status);
        if (boundary_status == true) // in bounds
          {
        	_workManager.PUSHNode(statusIt.GetIndex()
                    + m_NeighborList.GetNeighborhoodOffset(i), to);
          }
        }
      }
    }
  }
}
///////////////////////////////////////////////////////////////////////////////
template<class TInputImage,class TFeatureImage, class TOutputPixelType>
void
MySegmtLevelSetFilter_InitPart<TInputImage, TFeatureImage, TOutputPixelType>
::InitializeActiveLayerValues()
{
  const ValueType CHANGE_FACTOR = m_ConstantGradientValue / 2.0;
  ValueType MIN_NORM      = 1.0e-6;
  if (this->GetUseImageSpacing())
    {
    double minSpacing = NumericTraits<double>::max();
    for (unsigned int i=0; i<OutputImageType::ImageDimension; i++)
      {
      minSpacing = vnl_math_min(minSpacing,this->GetInput()->GetSpacing()[i]);
      }
    MIN_NORM *= minSpacing;
    }

  unsigned int i, center;

  typename LayerType::ConstIterator activeIt;
  ConstNeighborhoodIterator<OutputImageType>
    shiftedIt( m_NeighborList.GetRadius(), m_ShiftedImage,
               this->GetOutput()->GetRequestedRegion() );
  
  center = shiftedIt.Size() /2;
  typename OutputImageType::Pointer output = this->GetOutput();

  //const NeighborhoodScalesType neighborhoodScales; // = func_->ComputeNeighborhoodScales();

  ValueType dx_forward, dx_backward, length, distance;
  for(uint32 spuIt=0; spuIt<m_SPEManager.GetSPECount(); spuIt++)
    {
  // For all indicies in the active layer...
  for (activeIt = _workManager.GetLayers()[spuIt].layers[0]->Begin();
       activeIt != _workManager.GetLayers()[spuIt].layers[0]->End(); ++activeIt)
    {
    // Interpolate on the (shifted) input image values at this index to
    // assign an active layer value in the output image.
    shiftedIt.SetLocation( activeIt->m_Value );

    length = this->m_ValueZero;
    for (i = 0; i < OutputImageType::ImageDimension; ++i)
      {
      dx_forward = ( shiftedIt.GetPixel(center + m_NeighborList.GetStride(i))
        - shiftedIt.GetCenterPixel() );// * neighborhoodScales[i];
      dx_backward = ( shiftedIt.GetCenterPixel()
        - shiftedIt.GetPixel(center - m_NeighborList.GetStride(i)) );// * neighborhoodScales[i];

      if ( vnl_math_abs(dx_forward) > vnl_math_abs(dx_backward) )
        {
        length += dx_forward * dx_forward;
        }
      else
        {
        length += dx_backward * dx_backward;
        }
      }
    length = vcl_sqrt((double)length) + MIN_NORM;
    distance = shiftedIt.GetCenterPixel() / length;

    output->SetPixel( activeIt->m_Value , 
                      vnl_math_min(vnl_math_max(-CHANGE_FACTOR, distance), CHANGE_FACTOR) );
    }
    }
  
}

///////////////////////////////////////////////////////////////////////////////
template<class TInputImage,class TFeatureImage, class TOutputPixelType>
void
MySegmtLevelSetFilter_InitPart<TInputImage, TFeatureImage, TOutputPixelType>
::PostProcessOutput()
{
  // Assign background pixels INSIDE the sparse field layers to a new level set
  // with value less than the innermost layer.  Assign background pixels
  // OUTSIDE the sparse field layers to a new level set with value greater than
  // the outermost layer.
  const ValueType max_layer = static_cast<ValueType>(NUM_LAYERS);

  const ValueType inside_value  = (max_layer+1) * m_ConstantGradientValue;
  const ValueType outside_value = -(max_layer+1) * m_ConstantGradientValue;
 
  ImageRegionConstIterator<StatusImageType> statusIt(m_StatusImage,
                                                     this->GetOutput()->GetRequestedRegion());

  ImageRegionIterator<OutputImageType> outputIt(this->GetOutput(),
                                                this->GetOutput()->GetRequestedRegion());

  for (outputIt = outputIt.Begin(), statusIt = statusIt.Begin();
       ! outputIt.IsAtEnd(); ++outputIt, ++statusIt)
    {
    if (statusIt.Get() == this->m_StatusNull)
      {
      if (outputIt.Get() > this->m_ValueZero)
        {
        outputIt.Set(inside_value);
        }
      else
        {
        outputIt.Set(outside_value);
        }
      }
    }
}
///////////////////////////////////////////////////////////////////////////////

template <class TInputImage,class TFeatureImage, class TOutputPixelType>
void
MySegmtLevelSetFilter_InitPart<TInputImage, TFeatureImage, TOutputPixelType>
::PrintStats(std::ostream &s)
{
	s << "========= stats ===========" << std::endl;
	s << "Max. no. iterations: " << this->GetNumberOfIterations() << std::endl;
	s << "Max. RMS error: " << this->GetMaximumRMSError() << std::endl;
	s << "No. elpased iterations: " << this->GetElapsedIterations() << std::endl;
	s << "RMS change: " << this->GetRMSChange() << std::endl;
//	s << std::endl;
//	s << "Time spent in solver: " << cntr_ << std::endl;
//	s << "Time spent in difference solving: " << func_->cntr_ << std::endl;
	s << "===========================" << std::endl;
}

///////////////////////////////////////////////////////////////////////////////

}

#endif