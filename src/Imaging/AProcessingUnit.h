/**
 * @ingroup imaging 
 * @author Jan Kolomaznik 
 * @file AProcessingUnit.h 
 * @{ 
 **/

#ifndef _ABSTRACT_PROCESSING_UNIT_H
#define _ABSTRACT_PROCESSING_UNIT_H

#include <boost/shared_ptr.hpp>

/**
 *  @addtogroup imaging Imaging Library
 *  @{
 */

namespace M4D
{
namespace Imaging
{


/**
 *
 **/
class AProcessingUnit
{
public:
	/**
	 * Virtual destructor - to allow polymorphical destruction of successors.
	 **/
	virtual
	~AProcessingUnit() {}

protected:

private:

};

}/*namespace Imaging*/
}/*namespace M4D*/

/** @} */

#endif /*_ABSTRACT_PROCESSING_UNIT_H*/


/** @} */

