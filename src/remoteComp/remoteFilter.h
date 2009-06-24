/**
 * @ingroup cellbe 
 * @author Vaclav Klecanda 
 * @file RemoteFilter.h 
 * @{ 
 **/

#ifndef _REMOTE_FILTER_H
#define _REMOTE_FILTER_H

#include <sstream>
#include "netCommons.h"
#include "netAccessor.h"
#include "Imaging/AbstractImageFilterWholeAtOnce.h"
#include "remoteNodesManager.h"
#include "iRemoteFilterProperties.h"

namespace M4D
{
namespace RemoteComputing
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
  : public Imaging::AbstractImageFilterWholeAtOnce< InputImageType, OutputImageType >
  , public RemoteNodesManager
{
public:
	typedef typename Imaging::AbstractImageFilterWholeAtOnce< InputImageType, OutputImageType > PredecessorType;
	typedef typename PredecessorType::Properties Properties;
	
	RemoteFilter(iRemoteFilterProperties *p);
	~RemoteFilter();

protected:

	bool
	ProcessImage(
			const InputImageType 	&in,
			OutputImageType		&out
		    );
	
	void PrepareOutputDatasets(void);
	
  /**
   *  This method should count output image size based on filters
   *  that are in remote pipeline through SetImageSize() method
   *  of output ports.
   */
	//virtual void PrepareOutputDatasets() = 0;

private:	
	iRemoteFilterProperties *properties_;
	
	asio::io_service m_io_service;
	asio::ip::tcp::socket m_socket_;
	
	NetAccessor netAccessor_;
	
	void Connect(void);
	void Disconnect();
	void SendDataSet(void);
	bool RecieveDataSet(void);
	
	void SendCommand( eCommand command);
};

} /*namespace Imaging*/
} /*namespace M4D*/

//include implementation
#include "clientPart/remoteFilter.cxx"

#endif /*_REMOTE_FILTER_H*/

/** @} */

