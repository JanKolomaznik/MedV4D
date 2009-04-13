#ifndef CELLREADYTHRESHOLDSEGMENTATIONLEVELSETIMAGEFILTER_H_
#error File filter.tcc cannot be included directly!
#else

namespace itk {

///////////////////////////////////////////////////////////////////////////////


template<class TInputImage,class TFeatureImage, class TOutputPixelType>
MySegmtLevelSetFilter<TInputImage, TFeatureImage, TOutputPixelType>
::MySegmtLevelSetFilter()
{  
	this->m_Conf.m_upThreshold = 500;
	this->m_Conf.m_downThreshold = -500;
	this->m_Conf.m_propWeight = 1;
	this->m_Conf.m_curvWeight = 0.001f;
	this->m_Conf.m_ConstantGradientValue = this->m_ConstantGradientValue;
	
	applyUpdateCalc.m_layerGate.dispatcher = &this->m_requestDispatcher;
}
///////////////////////////////////////////////////////////////////////////////
template<class TInputImage,class TFeatureImage, class TOutputPixelType>
MySegmtLevelSetFilter<TInputImage, TFeatureImage, TOutputPixelType>
::~MySegmtLevelSetFilter()
{}


///////////////////////////////////////////////////////////////////////////////
//#if( ! (defined(COMPILE_FOR_CELL) || defined(COMPILE_ON_CELL) ) )
//
//template<class TInputImage,class TFeatureImage, class TOutputPixelType>
//void
//MySegmtLevelSetFilter<TInputImage, TFeatureImage, TOutputPixelType>
//::SetupGate()
//{
//  for(uint32 i=0; i<this->m_Layers.size() ; i++)
//  {
//	  m_gateLayerPointers[i] = (M4D::Cell::LayerGate::LayerType *) this->m_Layers[i].GetPointer();
//	  
//	  applyUpdateCalc.conf.layerBegins[i] = (M4D::Cell::SparseFieldLevelSetNode *) this->m_Layers[i]->Begin().GetPointer();
//	  applyUpdateCalc.conf.layerEnds[i] = (M4D::Cell::SparseFieldLevelSetNode *) this->m_Layers[i]->End().GetPointer();
//  }
//  applyUpdateCalc.SetGateProps(m_gateLayerPointers,
//		  (M4D::Cell::LayerGate::LayerNodeStorageType *)this->m_LayerNodeStore.GetPointer() );
//}
//#endif
///////////////////////////////////////////////////////////////////////////////

template<class TInputImage,class TFeatureImage, class TOutputPixelType>
void
MySegmtLevelSetFilter<TInputImage, TFeatureImage, TOutputPixelType>
::ApplyUpdate(TimeStepType dt)
{
#ifdef FOR_CELL
	  command = M4D::Cell::CALC_CHANGE;
	  m_SPEManager.SendCommand(command);
#else
	  //SetupGate();
	  uint32 i;
	    for(i=0; i<this->m_Layers.size() ; i++)
	    {	  	  
	  	  applyUpdateCalc.conf.layerBegins[i] = (M4D::Cell::SparseFieldLevelSetNode *) this->m_Layers[i]->Begin().GetPointer();
	  	  applyUpdateCalc.conf.layerEnds[i] = (M4D::Cell::SparseFieldLevelSetNode *) this->m_Layers[i]->End().GetPointer();
	    }
	  this->m_Conf.m_UpdateBufferData = &this->m_UpdateBuffer[0];
	  
	  applyUpdateCalc.SetCommonConfiguration(& this->m_Conf);
	  
//	    std::cout << "Update list:" << std::endl;
//	    PrintUpdateBuf(std::cout);
	  
	  this->SetRMSChange(applyUpdateCalc.ApplyUpdate(dt));
#endif	
}
///////////////////////////////////////////////////////////////////////////////
template<class TInputImage,class TFeatureImage, class TOutputPixelType>
void
MySegmtLevelSetFilter<TInputImage, TFeatureImage, TOutputPixelType>
::Initialize()
{
  this->InitializeInputAndConstructLayers();
	
  InitConfigStructures();
        
#if( defined(COMPILE_FOR_CELL) || defined(COMPILE_ON_CELL) )
	m_SPEManager.RunSPEs(&m_Conf);
#endif
  
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
template<class TInputImage,class TFeatureImage, class TOutputPixelType>
void
MySegmtLevelSetFilter<TInputImage, TFeatureImage, TOutputPixelType>
::InitConfigStructures(void)
{	        
	Superclass::InitConfigStructures();
}

///////////////////////////////////////////////////////////////////////////////
template<class TInputImage,class TFeatureImage, class TOutputPixelType>
typename
MySegmtLevelSetFilter<TInputImage, TFeatureImage, TOutputPixelType>::TimeStepType
MySegmtLevelSetFilter<TInputImage, TFeatureImage, TOutputPixelType>
::CalculateChange()
{
	this->AllocateUpdateBuffer();
	  
	  this->m_Conf.m_activeSetBegin = (M4D::Cell::SparseFieldLevelSetNode *) this->m_Layers[0]->Begin().GetPointer();
	  this->m_Conf.m_activeSetEnd = (M4D::Cell::SparseFieldLevelSetNode *) this->m_Layers[0]->End().GetPointer();
	  this->m_Conf.m_UpdateBufferData = &this->m_UpdateBuffer[0];
	  
	  TimeStepType dt;
	  
#ifdef FOR_CELL
	  command = M4D::Cell::CALC_CHANGE;
	  m_SPEManager.SendCommand(command);
#else
	  dt = Superclass::CalculateChange();
#endif	
	
	return dt;
}
///////////////////////////////////////////////////////////////////////////////

template<class TInputImage,class TFeatureImage, class TOutputPixelType>
void
MySegmtLevelSetFilter<TInputImage, TFeatureImage, TOutputPixelType>
::PropagateAllLayerValues()
{
	
#ifdef FOR_CELL
	  command = M4D::Cell::CALC_CHANGE;
	  m_SPEManager.SendCommand(command);
#else
	  Superclass::PropagateAllLayerValues();
#endif
}

///////////////////////////////////////////////////////////////////////////////
}
#endif
