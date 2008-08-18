#ifndef _REMOTE_FILTER_H
#define _REMOTE_FILTER_H

#include "Imaging/AbstractImageFilterWholeAtOnce.h"
#include "cellBE/RemoteFilterBase.h"
#include "cellBE/AbstractDataSetSerializer.h"

namespace M4D
{

namespace Imaging
{

/**
 *  Base class for every remote filter. 
 *  Remote filter contains definition
 *  of remote pipeline, that the filter represents. Actual definition has
 *  to be made through members of FilterProperties descendant type (for
 *  example ThresholdingFilter<>::Properties typed member represents
 *  thresholding filter). Current implementation lets only linear remote
 *  pipelines to be crated. Actual creation of pipeline is of course performed
 *  on the server but based on vector of FilterProperties that it recieves.
 *  So definig pipeline is through vector (defining vector) where order of filterProperties
 *  in vector corresponding with order of appropriate filter in remote
 *  pipeline. 
 *  FilterProperties are serialized through FilterPropertiesSerializers.
 *  So actual definig vector is vector of FilterPropertiesSerializers. This vector
 *  should be created in appropriate remote filter constructor and then passed
 *  to CreateJob method of CellClient object. Actual creation of 
 *  FilterPropertiesSerializers is done through templated methodes of 
 *  GeneralFilterSerializer object.
 *  Beside creation of FilterPropertiesSerializers there should be created
 *  apporopriate DataSetSerializers that performs actual dataSet serialization
 *  and deserialization of result dataSet sent from server. Both these object
 *  (input and output dataSetSerializers) has to passed as well to CreateJob method.
 */
template< typename InputImageType, typename OutputImageType >
class RemoteFilter 
	: public AbstractImageFilterWholeAtOnce< InputImageType, OutputImageType >
  , public M4D::CellBE::RemoteFilterBase
{
public:
	typedef typename  Imaging::AbstractImageFilterWholeAtOnce< InputImageType, OutputImageType > PredecessorType;
	typedef PredecessorType::Properties Properties;
	
	RemoteFilter( Properties *prop );
	~RemoteFilter();
protected:

	bool
	ProcessImage(
			const InputImageType 	&in,
			OutputImageType		&out
		    );

  /**
   *  This method should count output image size based on filters
   *  that are in remote pipeline through SetImageSize() method
   *  of output ports.
   */
	//virtual void PrepareOutputDatasets() = 0;

  // actual job responsible for actual work
  M4D::CellBE::ClientJob *m_job;

  M4D::CellBE::AbstractDataSetSerializer *m_inSerializer;
  M4D::CellBE::AbstractDataSetSerializer *m_outSerializer;

private:
	GET_PROPERTIES_DEFINITION_MACRO;

};

} /*namespace Imaging*/
} /*namespace M4D*/

//include implementation
#include "cellBE/RemoteFilter.tcc"

#endif /*_REMOTE_FILTER_H*/
