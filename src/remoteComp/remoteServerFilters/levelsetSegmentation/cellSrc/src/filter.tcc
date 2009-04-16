#ifndef CELLREADYTHRESHOLDSEGMENTATIONLEVELSETIMAGEFILTER_H_
#error File filter.tcc cannot be included directly!
#else

namespace itk {

///////////////////////////////////////////////////////////////////////////////


template<class TInputImage,class TFeatureImage, class TOutputPixelType>
MySegmtLevelSetFilter<TInputImage, TFeatureImage, TOutputPixelType>
::MySegmtLevelSetFilter()
{	
	this->applyUpdateCalc.m_layerGate.dispatcher = &this->m_requestDispatcher;
}
///////////////////////////////////////////////////////////////////////////////
template<class TInputImage,class TFeatureImage, class TOutputPixelType>
MySegmtLevelSetFilter<TInputImage, TFeatureImage, TOutputPixelType>
::~MySegmtLevelSetFilter()
{}

///////////////////////////////////////////////////////////////////////////////
template<class TInputImage,class TFeatureImage, class TOutputPixelType>
void
MySegmtLevelSetFilter<TInputImage, TFeatureImage, TOutputPixelType>
::Initialize()
{
  this->InitializeInputAndConstructLayers();
	
  this->InitRunConf();
        
#ifdef FOR_CELL
	m_SPEManager.RunSPEs(&m_Conf);
#else
	// setup apply update
	this->applyUpdateCalc.commonConf = & this->m_conf.runConf;
	this->applyUpdateCalc.m_stepConfig = & this->m_conf.calcChngApplyUpdateConf;	
	this->applyUpdateCalc.m_propLayerValuesConfig = & this->m_conf.propagateValsConf;	
	
	// and update solver
	this->updateSolver.m_Conf = & this->m_conf.runConf;
	this->updateSolver.m_stepConfig = & this->m_conf.calcChngApplyUpdateConf;
	this->updateSolver.Init();
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
::ApplyUpdate(TimeStepType dt)
{
	this->InitCalculateChangeAndUpdActiveLayerConf();
	this->InitPropagateValuesConf();
	
	  LOG("================= ApplyUpdate	===================================================");
	  	  		  	for(uint32 i=0; i<LYERCOUNT; i++)
	  	  		  		  	  LOG("Layer " << i << "size: " <<	this->m_Layers[i]->Size());
	  	  		  	
#ifdef FOR_CELL
	  command = M4D::Cell::CALC_CHANGE;
	  m_SPEManager.SendCommand(command);
#else	  
	  Superclass::ApplyUpdate(dt);
	  
	  LOG("after");
	  	  	  	  for(uint32 i=0; i<LYERCOUNT; i++)
	  	  	  	  		  		  	  LOG("Layer " << i << "size: " <<	this->m_Layers[i]->Size());
#endif	
}

///////////////////////////////////////////////////////////////////////////////
template<class TInputImage,class TFeatureImage, class TOutputPixelType>
typename
MySegmtLevelSetFilter<TInputImage, TFeatureImage, TOutputPixelType>::TimeStepType
MySegmtLevelSetFilter<TInputImage, TFeatureImage, TOutputPixelType>
::CalculateChange()
{
	this->AllocateUpdateBuffer();
	this->InitCalculateChangeAndUpdActiveLayerConf();
	  
	  TimeStepType dt;
	  

	  LOG("================= Calc change	  ===================================================");
	  LOG("Layer0 " << "size: " << this->m_Layers[0]->Size());


	 	 	  	this->PrintUpdateBuf(LOUT);

	  
#ifdef FOR_CELL
	  command = M4D::Cell::CALC_CHANGE;
	  m_SPEManager.SendCommand(command);
#else
	  dt = Superclass::CalculateChange();
#endif	
	  

	  LOG("Layer0 after" << "size: " << this->m_Layers[0]->Size());

	 	 	  	  LOG("dt=" << dt);

	 	 	  	this->PrintUpdateBuf(LOUT);
	
	return dt;
}
///////////////////////////////////////////////////////////////////////////////

template<class TInputImage,class TFeatureImage, class TOutputPixelType>
void
MySegmtLevelSetFilter<TInputImage, TFeatureImage, TOutputPixelType>
::PropagateAllLayerValues()
{
	this->InitPropagateValuesConf();
	
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
