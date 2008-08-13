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

  // filter serializer vector that will define actual remote pipeline
  FilterSerializerVector m_filterSerializers;

  // put into the vector serializers instances in order that is in remote pipe
  ser = GeneralFilterSerializer::GetFilterSerializer<ThresholdingOptsType>()
  m_filterSerializers.push_back( ser);

  // create dataSetSerializers
  AbstractDataSetSerializer *inSerializer = NULL;
  AbstractDataSetSerializer *outerializer = NULL;
  GeneralDataSetSerializer::GetSerializer<InType>(getInput().GetDataSet());
  GeneralDataSetSerializer::GetSerializer<OutType>(getOutPut().GetDataSet());

  // create job
  m_job = m_cellClient.CreateJob( m_filterSerializers, inSerializer, outSerializer);
}

///////////////////////////////////////////////////////////////////////////////

template< typename InType, typename OutType>
void 
BoneSegmentationRemote<InType, OutType>::PrepareOutputDatasets()
{
  AbstractDataSet *in, *out;

  // count output dataSet size according inner filters and set it
  int size = 0;
  getOutPort().SetImageSize( size);

  m_job->SetDataSets( in, out);
}

///////////////////////////////////////////////////////////////////////////////

} /*namespace Imaging*/
} /*namespace M4D*/

#endif
