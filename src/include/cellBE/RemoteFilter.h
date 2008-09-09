/**
 * @ingroup cellbe 
 * @author Vaclav Klecanda 
 * @file RemoteFilter.h 
 * @{ 
 **/

#ifndef _REMOTE_FILTER_H
#define _REMOTE_FILTER_H

#include "Imaging/AbstractImageFilterWholeAtOnce.h"
#include "cellBE/RemoteFilterBase.h"
#include "cellBE/AbstractDataSetSerializer.h"
#include "cellBE/AbstractFilterSerializer.h"
#include "cellBE/FilterSerializerArray.h"

namespace M4D
{

namespace Imaging
{

/**
 * Macro unwinding to get method for property.
 * \param TYPE Type of property - return value of the method.
 * \param NAME Name of property used in name of function - Get'NAME'().
 * \param \PROPERTY_NAME Name of property in Properties structure.
 **/
#define GET_REMOTE_PROPERTY_METHOD_MACRO( TYPE, NAME, PROPERTY_NAME, PROPERTIES, PROPERTIES_TYPE ) \
	TYPE Get##NAME ()const{ return (static_cast<PROPERTIES_TYPE&>( this->PROPERTIES ) ).PROPERTY_NAME ; }

/**
 * Macro unwinding to set method for property.
 * \param TYPE Type of property - parameter type of the method.
 * \param NAME Name of property used in name of function - Set'NAME'().
 * \param \PROPERTY_NAME Name of property in Properties structure.
 **/
#define SET_REMOTE_PROPERTY_METHOD_MACRO( TYPE, NAME, PROPERTY_NAME, PROPERTIES, PROPERTIES_TYPE ) \
	void Set##NAME ( TYPE value ){ this->_properties->IncTimestamp(); (static_cast<PROPERTIES_TYPE&>( this->PROPERTIES ) ).PROPERTY_NAME = value; }

/**
 * Macro unwinding to previously defined macros.
 **/
#define GET_SET_REMOTE_PROPERTY_METHOD_MACRO( TYPE, NAME, PROPERTY_NAME, PROPERTIES, PROPERTIES_TYPE ) \
	GET_REMOTE_PROPERTY_METHOD_MACRO( TYPE, NAME, PROPERTY_NAME, PROPERTIES, PROPERTIES_TYPE ) \
	SET_REMOTE_PROPERTY_METHOD_MACRO( TYPE, NAME, PROPERTY_NAME, PROPERTIES, PROPERTIES_TYPE ) 


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
	typedef typename Imaging::AbstractImageFilterWholeAtOnce< InputImageType, OutputImageType > PredecessorType;
	typedef typename PredecessorType::Properties Properties;
	
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

/** @} */

