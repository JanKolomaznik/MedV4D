#ifndef BONE_SEGMENTATION_FILTER_H
#error File BoneSegmentationRemote.tcc cannot be included directly!
#else

namespace M4D
{
namespace Imaging
{

///////////////////////////////////////////////////////////////////////////////

template< typename InType, typename OutType>
BoneSegmentationRemote<InType, OutType>::BoneSegmentationRemote()
{
  AbstractFilterSerializer *ser;  

  // put into the vector serializers instances in order that is in remote pipe
  //ser = GeneralFilterSerializer::CreateSerializer<ThresholdingOptsType>()
  m_filterSerializers.push_back( ser);
}

///////////////////////////////////////////////////////////////////////////////

template< typename InType, typename OutType>
void 
BoneSegmentationRemote<InType, OutType>::PrepareOutputDatasets()
{
  // count output dataSet size according inner filters and set it
  int size = 0;
  getOutPort().SetImageSize( size);

  // create job
  m_job = m_cellClient.CreateJob( m_filterSerializers, getInput().GetDataSet());
}

///////////////////////////////////////////////////////////////////////////////

} /*namespace Imaging*/
} /*namespace M4D*/

#endif
