/**
 * @ingroup cellbe 
 * @author Vaclav Klecanda 
 * @file SimpleMIPRemote.h
 * @{ 
 **/

#ifndef SIMPLE_MIP_REMOTE_H
#define SIMPLE_MIP_REMOTE_H

#include "cellBE/RemoteFilter.h"


// include needed filters ...
#include "Imaging/filters/ThresholdingFilter.h"
#include "Imaging/filters/MedianFilter.h"

namespace M4D
{
namespace Imaging
{

/**
 *  This is the firs exemplar of remote filter. It contains specification
 *  of an remote pipeline, that contains few filters that should perform
 *  bone segmentaion.
 *  This is example how remote filter currntly works and another remote
 *  filter should be written in this manner.
 *  Basic rule is to inherit from RemoteFilter templated class.
 */
template< typename ImageType >
class SimpleMIPRemote
  : public RemoteFilter<ImageType, ImageType>
{
public:
	typedef typename RemoteFilter<ImageType, ImageType> PredecessorType;
	typedef PredecessorType::Properties	Properties;

	SimpleMIPRemote();
  

	/////////////////// To customize /////////////////////
	// puting options available to outer world to be able to specify it ....

  // thresholding filter issues
	typedef ThresholdingFilter<ImageType>	SimpleMIP;
	typedef typename SimpleMIP::Properties SimpleMIPOptsType;
	
	SimpleMIPOptsType *GetThreshholdingOptions( void)	
  {
		return &m_simpleMIPOptions;
	}


protected:
	void PrepareOutputDatasets();

private:

	/**
	 * Here should be added members of  filter options type that will
	 * define the remote pipeline this filter represents. Each member
	 * for single filter in remote pipeline. As a next step is defining
	 * retrieving public members, that will provide ability to change
	 * the filter options from outer world.
	 **/
  SimpleMIPOptsType m_simpleMIPOptions;
	// ...


	//////////////////////////////////////////////////////

};

} /*namespace Imaging*/
} /*namespace M4D*/

//include implementation
#include "cellBE/remoteFilters/SimpleMIPRemote.tcc"

#endif

/** @} */

