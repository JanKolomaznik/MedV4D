#ifndef _ABSTRACT_PROCESSING_UNIT_H
#define _ABSTRACT_PROCESSING_UNIT_H

#include <boost/shared_ptr.hpp>

namespace M4D
{
namespace Imaging
{


/**
 *
 **/
class AbstractProcessingUnit
{
public:
	virtual
	~AbstractProcessingUnit() {}

	virtual void
	Execute()=0;

	virtual bool
	StopExecution()=0;
protected:

private:

};

}/*namespace Imaging*/
}/*namespace M4D*/


#endif /*_ABSTRACT_PROCESSING_UNIT_H*/

