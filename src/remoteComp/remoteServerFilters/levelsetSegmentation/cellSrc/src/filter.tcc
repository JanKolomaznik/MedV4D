#ifndef CELLREADYTHRESHOLDSEGMENTATIONLEVELSETIMAGEFILTER_H_
#error File filter.tcc cannot be included directly!
#else

namespace itk {

///////////////////////////////////////////////////////////////////////////////

template<class TInputImage,class TFeatureImage, class TOutputPixelType>
MySegmtLevelSetFilter<TInputImage, TFeatureImage, TOutputPixelType>
::MySegmtLevelSetFilter()
{
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
  
  this->m_SPEManager.InitProgramProps();
        
#ifdef FOR_CELL
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
::ApplyUpdate(TimeStepType dt)
{	  	  		  	
//#ifdef FOR_CELL
//	  command = M4D::Cell::CALC_CHANGE;
//	  m_SPEManager.SendCommand(command);
//#else	  
//	  Superclass::ApplyUpdate(dt);
//#endif
	this->SetRMSChange(this->m_SPEManager.ApplyUpdate(dt));	
}

///////////////////////////////////////////////////////////////////////////////
template<class TInputImage,class TFeatureImage, class TOutputPixelType>
typename
MySegmtLevelSetFilter<TInputImage, TFeatureImage, TOutputPixelType>::TimeStepType
MySegmtLevelSetFilter<TInputImage, TFeatureImage, TOutputPixelType>
::CalculateChange()
{	  
	  TimeStepType dt;
	  
	  //this->_workManager.PrintLists(LOUT);

//#ifdef FOR_CELL
//	  command = M4D::Cell::CALC_CHANGE;
//	  m_SPEManager.SendCommand(command);
//#else
	  dt = this->m_SPEManager.RunUpdateCalc();
//#endif	
	
	return dt;
}
///////////////////////////////////////////////////////////////////////////////

template<class TInputImage,class TFeatureImage, class TOutputPixelType>
void
MySegmtLevelSetFilter<TInputImage, TFeatureImage, TOutputPixelType>
::PropagateAllLayerValues()
{
//	this->InitPropagateValuesConf();
//	
//#ifdef FOR_CELL
//	  command = M4D::Cell::CALC_CHANGE;
//	  m_SPEManager.SendCommand(command);
//#else
//	  Superclass::PropagateAllLayerValues();
//#endif
	
	this->m_SPEManager.RunPropagateLayerVals();
}

///////////////////////////////////////////////////////////////////////////////
}
#endif
