#ifndef CELLREADYTHRESHOLDSEGMENTATIONLEVELSETIMAGEFILTER_H_
#error File filter.tcc cannot be included directly!
#else

namespace itk {

///////////////////////////////////////////////////////////////////////////////

template <class TInputImage,class TFeatureImage, class TOutputPixelType>
ThreshSegLevelSetFilter<TInputImage, TFeatureImage, TOutputPixelType>
::ThreshSegLevelSetFilter(void)
{	  
	m_IsoSurfaceValue = m_ValueZero;
    m_NumberOfLayers = ImageDimension;
    m_LayerNodeStore = LayerNodeStorageType::New();
    m_LayerNodeStore->SetGrowthStrategyToExponential();
    this->SetRMSChange(static_cast<double>(m_ValueZero));
    m_BoundsCheckingActive = false;
    m_ConstantGradientValue = 1.0;
	  
  // Provide some reasonable defaults which will at least prevent infinite
  // looping.
  this->SetMaximumRMSError(0.02);
  this->SetNumberOfIterations(1000);
}

///////////////////////////////////////////////////////////////////////////////

template<class TInputImage, class TOutputImage>
void
SparseFieldLevelSetImageFilter<TInputImage, TOutputImage>
::ApplyUpdate(TimeStepType dt)
{
  // zkontrolovat pripadne predelat delku linkd listu, tak aby kazde SPU melo stejne prace
}

///////////////////////////////////////////////////////////////////////////////
template<class TInputImage, class TOutputImage>
void
ThreshSegLevelSetFilter<TInputImage, TOutputImage>
::CopyInputToOutput()
{
  // This method is the first step in initializing the level-set image, which
  // is also the output of the filter.  The input is passed through a
  // zero crossing filter, which produces zero's at pixels closest to the zero
  // level set and one's elsewhere.  The actual zero level set values will be
  // adjusted in the Initialize() step to more accurately represent the
  // position of the zero level set.

  // First need to subtract the iso-surface value from the input image.
  typedef ShiftScaleImageFilter<InputImageType, OutputImageType> ShiftScaleFilterType;
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
  zeroCrossingFilter->SetBackgroundValue(m_ValueOne);
  zeroCrossingFilter->SetForegroundValue(m_ValueZero);

  zeroCrossingFilter->Update();

  this->GraftOutput(zeroCrossingFilter->GetOutput());
}
///////////////////////////////////////////////////////////////////////////////  
template<class TInputImage, class TOutputImage>
void
ThreshSegLevelSetFilter<TInputImage, TOutputImage>
::Initialize()
{
  unsigned int i;

  if (this->GetUseImageSpacing())
    {
    double minSpacing = NumericTraits<double>::max();
    for (i=0; i<ImageDimension; i++)
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
    statusIt.Set( m_StatusNull );
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
      statusIt.Set( m_StatusBoundaryPixel );
      }
    }

  // Erase all existing layer lists.
  for (i = 0; i < m_Layers.size(); ++i)
    {
    while (! m_Layers[i]->Empty() )
      {
      m_LayerNodeStore->Return(m_Layers[i]->Front());
      m_Layers[i]->PopFront();
      }
    }
  
  // Allocate the layers for the sparse field.
  m_Layers.clear();
  m_Layers.reserve(2*m_NumberOfLayers + 1);

  while ( m_Layers.size() < (2*m_NumberOfLayers+1) )
    {
    m_Layers.push_back( LayerType::New() );
    }
  
  // Throw an exception if we don't have enough layers.
  if (m_Layers.size() < 3)
    {
    itkExceptionMacro( << "Not enough layers have been allocated for the sparse field.  Requires at least one layer.");
    }
  
  // Construct the active layer and initialize the first layers inside and
  // outside of the active layer.
  this->ConstructActiveLayer();

  // Construct the rest of the non-active set layers using the first two
  // layers. Inside layers are odd numbers, outside layers are even numbers.
  for (i = 1; i < m_Layers.size() - 2; ++i)
    {
    this->ConstructLayer(i, i+2);
    }
  
  // Set the values in the output image for the active layer.
  this->InitializeActiveLayerValues();
 
  // Initialize layer values using the active layer as seeds.
  this->PropagateAllLayerValues();

  // Initialize pixels inside and outside the sparse field layers to positive
  // and negative values, respectively.  This is not necessary for the
  // calculations, but is useful for presenting a more intuitive output to the
  // filter.  See PostProcessOutput method for more information.
  this->InitializeBackgroundPixels();
}
///////////////////////////////////////////////////////////////////////////////
template <class TInputImage, class TOutputImage>
void
ThreshSegLevelSetFilter<TInputImage, TOutputImage>
::InitializeBackgroundPixels()
{
  // Assign background pixels OUTSIDE the sparse field layers to a new level set
  // with value greater than the outermost layer.  Assign background pixels
  // INSIDE the sparse field layers to a new level set with value less than
  // the innermost layer.
  const ValueType max_layer = static_cast<ValueType>(m_NumberOfLayers);

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
    if (statusIt.Get() == m_StatusNull || statusIt.Get() == m_StatusBoundaryPixel)
      {
      if (shiftedIt.Get() > m_ValueZero)
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
template <class TInputImage, class TOutputImage>
void
ThreshSegLevelSetFilter<TInputImage, TOutputImage>
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
  LayerNodeType *node;
  bool bounds_status;
  ValueType value;
  StatusType layer_number;

  typename OutputImageType::IndexType upperBounds, lowerBounds;
  lowerBounds = this->GetOutput()->GetRequestedRegion().GetIndex();
  upperBounds = this->GetOutput()->GetRequestedRegion().GetIndex()
    + this->GetOutput()->GetRequestedRegion().GetSize();

  for (outputIt.GoToBegin(); !outputIt.IsAtEnd(); ++outputIt)
    {
    if ( outputIt.GetCenterPixel() == m_ValueZero )
      {
      // Grab the neighborhood in the status image.
      center_index = outputIt.GetIndex();
      statusIt.SetLocation( center_index );

      // Check to see if any of the sparse field touches a boundary.  If so,
      // then activate bounds checking.
      for (i = 0; i < ImageDimension; i++)
        {
        if (center_index[i] + static_cast<long>(m_NumberOfLayers) >= (upperBounds[i] - 1)
            || center_index[i] - static_cast<long>(m_NumberOfLayers) <= lowerBounds[i])
          {
          m_BoundsCheckingActive = true;
          }
        }
      
      // Borrow a node from the store and set its value.
      node = m_LayerNodeStore->Borrow();
      node->m_Value = center_index;

      // Add the node to the active list and set the status in the status
      // image.
      m_Layers[0]->PushFront( node );
      statusIt.SetCenterPixel( 0 );

      // Grab the neighborhood in the image of shifted input values.
      shiftedIt.SetLocation( center_index );
          
      // Search the neighborhood pixels for first inside & outside layer
      // members.  Construct these lists and set status list values. 
      for (i = 0; i < m_NeighborList.GetSize(); ++i)
        {
        offset_index = center_index
          + m_NeighborList.GetNeighborhoodOffset(i);

        if ( outputIt.GetPixel(m_NeighborList.GetArrayIndex(i)) != m_ValueZero)
          {
          value = shiftedIt.GetPixel(m_NeighborList.GetArrayIndex(i));

          if ( value < m_ValueZero ) // Assign to first inside layer.
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
            node = m_LayerNodeStore->Borrow();
            node->m_Value = offset_index;
            m_Layers[layer_number]->PushFront( node );
            } // else do nothing.
          }
        }
      }
    }
}
///////////////////////////////////////////////////////////////////////////////
template<class TInputImage, class TOutputImage>
void
ThreshSegLevelSetFilter<TInputImage, TOutputImage>
::ConstructLayer(StatusType from, StatusType to)
{
  unsigned int i;
  LayerNodeType *node;
  bool boundary_status;
  typename LayerType::ConstIterator fromIt;
  NeighborhoodIterator<StatusImageType>
    statusIt(m_NeighborList.GetRadius(), m_StatusImage,
             this->GetOutput()->GetRequestedRegion() );
  
  // For all indicies in the "from" layer...
  for (fromIt = m_Layers[from]->Begin();
       fromIt != m_Layers[from]->End();  ++fromIt)
    {
    // Search the neighborhood of this index in the status image for
    // unassigned indicies. Push those indicies onto the "to" layer and
    // assign them values in the status image.  Status pixels outside the
    // boundary will be ignored.
    statusIt.SetLocation( fromIt->m_Value );
    for (i = 0; i < m_NeighborList.GetSize(); ++i)
      {
      if ( statusIt.GetPixel( m_NeighborList.GetArrayIndex(i) )
           == m_StatusNull )
        {
        statusIt.SetPixel(m_NeighborList.GetArrayIndex(i), to,
                          boundary_status);
        if (boundary_status == true) // in bounds
          {
          node = m_LayerNodeStore->Borrow();
          node->m_Value = statusIt.GetIndex()
            + m_NeighborList.GetNeighborhoodOffset(i);
          m_Layers[to]->PushFront( node );
          }
        }
      }
    }
}
///////////////////////////////////////////////////////////////////////////////
template <class TInputImage, class TOutputImage>
void
ThreshSegLevelSetFilter<TInputImage, TOutputImage>
::InitializeActiveLayerValues()
{
  const ValueType CHANGE_FACTOR = m_ConstantGradientValue / 2.0;
  ValueType MIN_NORM      = 1.0e-6;
  if (this->GetUseImageSpacing())
    {
    double minSpacing = NumericTraits<double>::max();
    for (unsigned int i=0; i<ImageDimension; i++)
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

  const NeighborhoodScalesType neighborhoodScales = this->GetDifferenceFunction()->ComputeNeighborhoodScales();

  ValueType dx_forward, dx_backward, length, distance;

  // For all indicies in the active layer...
  for (activeIt = m_Layers[0]->Begin();
       activeIt != m_Layers[0]->End(); ++activeIt)
    {
    // Interpolate on the (shifted) input image values at this index to
    // assign an active layer value in the output image.
    shiftedIt.SetLocation( activeIt->m_Value );

    length = m_ValueZero;
    for (i = 0; i < ImageDimension; ++i)
      {
      dx_forward = ( shiftedIt.GetPixel(center + m_NeighborList.GetStride(i))
        - shiftedIt.GetCenterPixel() ) * neighborhoodScales[i];
      dx_backward = ( shiftedIt.GetCenterPixel()
        - shiftedIt.GetPixel(center - m_NeighborList.GetStride(i)) ) * neighborhoodScales[i];

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
///////////////////////////////////////////////////////////////////////////////
template <class TInputImage, class TOutputImage>
void
ThreshSegLevelSetFilter<TInputImage, TOutputImage>
::AllocateUpdateBuffer()
{
  // Preallocate the update buffer.  NOTE: There is currently no way to
  // downsize a std::vector. This means that the update buffer will grow
  // dynamically but not shrink.  In newer implementations there may be a
  // squeeze method which can do this.  Alternately, we can implement our own
  // strategy for downsizing.
  m_UpdateBuffer.clear();
  m_UpdateBuffer.reserve(m_Layers[0]->Size());
}

///////////////////////////////////////////////////////////////////////////////
template <class TInputImage, class TOutputImage>
typename
ThreshSegLevelSetFilter<TInputImage, TOutputImage>::TimeStepType
ThreshSegLevelSetFilter<TInputImage, TOutputImage>
::CalculateChange()
{
  const typename Superclass::FiniteDifferenceFunctionType::Pointer df
    = this->GetDifferenceFunction();
  typename Superclass::FiniteDifferenceFunctionType::FloatOffsetType offset;
  ValueType norm_grad_phi_squared, dx_forward, dx_backward, forwardValue,
    backwardValue, centerValue;
  unsigned i;
  ValueType MIN_NORM      = 1.0e-6;
  if (this->GetUseImageSpacing())
    {
    double minSpacing = NumericTraits<double>::max();
    for (i=0; i<ImageDimension; i++)
      {
      minSpacing = vnl_math_min(minSpacing,this->GetInput()->GetSpacing()[i]);
      }
    MIN_NORM *= minSpacing;
    }

  void *globalData = df->GetGlobalDataPointer();
  
  typename LayerType::ConstIterator layerIt;
  NeighborhoodIterator<OutputImageType> outputIt(df->GetRadius(),
                    this->GetOutput(), this->GetOutput()->GetRequestedRegion());
  TimeStepType timeStep;

  const NeighborhoodScalesType neighborhoodScales = this->GetDifferenceFunction()->ComputeNeighborhoodScales();

  if ( m_BoundsCheckingActive == false )
    {
    outputIt.NeedToUseBoundaryConditionOff();
    }
  
  m_UpdateBuffer.clear();
  m_UpdateBuffer.reserve(m_Layers[0]->Size());

  // Calculates the update values for the active layer indicies in this
  // iteration.  Iterates through the active layer index list, applying 
  // the level set function to the output image (level set image) at each
  // index.  Update values are stored in the update buffer.
  for (layerIt = m_Layers[0]->Begin(); layerIt != m_Layers[0]->End(); ++layerIt)
    {
    outputIt.SetLocation(layerIt->m_Value);

    // Calculate the offset to the surface from the center of this
    // neighborhood.  This is used by some level set functions in sampling a
    // speed, advection, or curvature term.
    if(centerValue = outputIt.GetCenterPixel()) != 0.0 )
      {
      // Surface is at the zero crossing, so distance to surface is:
      // phi(x) / norm(grad(phi)), where phi(x) is the center of the
      // neighborhood.  The location is therefore
      // (i,j,k) - ( phi(x) * grad(phi(x)) ) / norm(grad(phi))^2
      norm_grad_phi_squared = 0.0;
      for (i = 0; i < ImageDimension; ++i)
        {
        forwardValue  = outputIt.GetNext(i);
        backwardValue = outputIt.GetPrevious(i);
            
        if (forwardValue * backwardValue >= 0)
          { //  Neighbors are same sign OR at least one neighbor is zero.
          dx_forward  = forwardValue - centerValue;
          dx_backward = centerValue - backwardValue;

          // Pick the larger magnitude derivative.
          if (::vnl_math_abs(dx_forward) > ::vnl_math_abs(dx_backward) )
            {
            offset[i] = dx_forward;
            }
          else
            {
            offset[i] = dx_backward;
            }
          }
        else //Neighbors are opposite sign, pick the direction of the 0 surface.
          {
          if (forwardValue * centerValue < 0)
            {
            offset[i] = forwardValue - centerValue;
            }
          else
            {
            offset[i] = centerValue - backwardValue;
            }
          }
        
        norm_grad_phi_squared += offset[i] * offset[i];
        }
      
      for (i = 0; i < ImageDimension; ++i)
        {
        offset[i] = (offset[i] * centerValue) / (norm_grad_phi_squared + MIN_NORM);
        }
          
      m_UpdateBuffer.push_back( df->ComputeUpdate(outputIt, globalData, offset) );
      }
    }
  
  // Ask the finite difference function to compute the time step for
  // this iteration.  We give it the global data pointer to use, then
  // ask it to free the global data memory.
  timeStep = df->ComputeGlobalTimeStep(globalData);

  df->ReleaseGlobalDataPointer(globalData);
  
  return timeStep;
}

///////////////////////////////////////////////////////////////////////////////
template <class TInputImage, class TOutputImage>
void
ThreshSegLevelSetFilter<TInputImage, TOutputImage>
::PropagateAllLayerValues()
{
  unsigned int i;

  // Update values in the first inside and first outside layers using the
  // active layer as a seed. Inside layers are odd numbers, outside layers are
  // even numbers. 
  this->PropagateLayerValues(0, 1, 3, 1); // first inside
  this->PropagateLayerValues(0, 2, 4, 2); // first outside

  // Update the rest of the layers.
  for (i = 1; i < m_Layers.size() - 2; ++i)
    {
    this->PropagateLayerValues(i, i+2, i+4, (i+2)%2);
    }
}

///////////////////////////////////////////////////////////////////////////////

template <class TInputImage, class TOutputImage>
void
ThreshSegLevelSetFilter<TInputImage, TOutputImage>
::PropagateLayerValues(StatusType from, StatusType to,
                       StatusType promote, int InOrOut)
{
  unsigned int i;
  ValueType value, value_temp, delta;
  value = NumericTraits<ValueType>::Zero; // warnings
  bool found_neighbor_flag;
  typename LayerType::Iterator toIt;
  LayerNodeType *node;
  StatusType past_end = static_cast<StatusType>( m_Layers.size() ) - 1;
  
  // Are we propagating values inward (more negative) or outward (more
  // positive)?
  if (InOrOut == 1) delta = - m_ConstantGradientValue;
  else delta = m_ConstantGradientValue;
 
  NeighborhoodIterator<OutputImageType>
    outputIt(m_NeighborList.GetRadius(), this->GetOutput(),
             this->GetOutput()->GetRequestedRegion() );
  NeighborhoodIterator<StatusImageType>
    statusIt(m_NeighborList.GetRadius(), m_StatusImage,
             this->GetOutput()->GetRequestedRegion() );

  if ( m_BoundsCheckingActive == false )
    {
    outputIt.NeedToUseBoundaryConditionOff();
    statusIt.NeedToUseBoundaryConditionOff();
    }
  
  toIt  = m_Layers[to]->Begin();
  while ( toIt != m_Layers[to]->End() )
    {
    statusIt.SetLocation( toIt->m_Value );

    // Is this index marked for deletion? If the status image has
    // been marked with another layer's value, we need to delete this node
    // from the current list then skip to the next iteration.
    if (statusIt.GetCenterPixel() != to)
      {
      node = toIt.GetPointer();
      ++toIt;
      m_Layers[to]->Unlink( node );
      m_LayerNodeStore->Return( node );
      continue;
      }
      
    outputIt.SetLocation( toIt->m_Value );

    found_neighbor_flag = false;
    for (i = 0; i < m_NeighborList.GetSize(); ++i)
      {
      // If this neighbor is in the "from" list, compare its absolute value
      // to to any previous values found in the "from" list.  Keep the value
      // that will cause the next layer to be closest to the zero level set.
      if ( statusIt.GetPixel( m_NeighborList.GetArrayIndex(i) ) == from )
        {
        value_temp = outputIt.GetPixel( m_NeighborList.GetArrayIndex(i) );

        if (found_neighbor_flag == false)
          {
          value = value_temp;
          }
        else
          {
          if (InOrOut == 1)
            {
            // Find the largest (least negative) neighbor
            if ( value_temp > value )
              {
              value = value_temp;
              }
            }
          else
            {
            // Find the smallest (least positive) neighbor
            if (value_temp < value)
              {
              value = value_temp;
              }
            }
          }
        found_neighbor_flag = true;
        }
      }
    if (found_neighbor_flag == true)
      {
      // Set the new value using the smallest distance
      // found in our "from" neighbors.
      outputIt.SetCenterPixel( value + delta );
      ++toIt;
      }
    else
      {
      // Did not find any neighbors on the "from" list, then promote this
      // node.  A "promote" value past the end of my sparse field size
      // means delete the node instead.  Change the status value in the
      // status image accordingly.
      node  = toIt.GetPointer();
      ++toIt;
      m_Layers[to]->Unlink( node );
      if ( promote > past_end )
        {
        m_LayerNodeStore->Return( node );
        statusIt.SetCenterPixel(m_StatusNull);
        }
      else
        {
        m_Layers[promote]->PushFront( node );
        statusIt.SetCenterPixel(promote);
        }
      }
    }
}

///////////////////////////////////////////////////////////////////////////////

template<class TInputImage, class TOutputImage>
void
ThreshSegLevelSetFilter<TInputImage, TOutputImage>
::PostProcessOutput()
{
  // Assign background pixels INSIDE the sparse field layers to a new level set
  // with value less than the innermost layer.  Assign background pixels
  // OUTSIDE the sparse field layers to a new level set with value greater than
  // the outermost layer.
  const ValueType max_layer = static_cast<ValueType>(m_NumberOfLayers);

  const ValueType inside_value  = (max_layer+1) * m_ConstantGradientValue;
  const ValueType outside_value = -(max_layer+1) * m_ConstantGradientValue;
 
  ImageRegionConstIterator<StatusImageType> statusIt(m_StatusImage,
                                                     this->GetOutput()->GetRequestedRegion());

  ImageRegionIterator<OutputImageType> outputIt(this->GetOutput(),
                                                this->GetOutput()->GetRequestedRegion());

  for (outputIt = outputIt.Begin(), statusIt = statusIt.Begin();
       ! outputIt.IsAtEnd(); ++outputIt, ++statusIt)
    {
    if (statusIt.Get() == m_StatusNull)
      {
      if (outputIt.Get() > m_ValueZero)
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
ThreshSegLevelSetFilter<TInputImage, TFeatureImage, TOutputPixelType>
::PrintStats(std::ostream &s)
{
	s << "========= stats ===========" << std::endl;
	s << "Max. no. iterations: " << this->GetNumberOfIterations() << std::endl;
	s << "Max. RMS error: " << this->GetMaximumRMSError() << std::endl;
	s << "No. elpased iterations: " << this->GetElapsedIterations() << std::endl;
	s << "RMS change: " << this->GetRMSChange() << std::endl;
	s << std::endl;
	s << "Time spent in solver: " << cntr_ << std::endl;
	s << "Time spent in CalculateChange: " << this->cntrCalculateChange_ << std::endl;
	s << "Time spent in ApplyUpdate_: " << this->cntrApplyUpdate_ << std::endl;
	s << "Time spent in difference solving: " << func_->cntr_ << std::endl;
	s << "===========================" << std::endl;
}

///////////////////////////////////////////////////////////////////////////////
}
#endif
