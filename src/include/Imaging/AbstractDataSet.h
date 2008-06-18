#ifndef _ABSTRACT_DATA_SET_H
#define _ABSTRACT_DATA_SET_H

#include <boost/shared_ptr.hpp>
#include "TimeStamp.h"

namespace M4D
{
namespace Imaging
{


class AbstractDataSet
{
public:
	/**
	 * Smart pointer to AbstractDataSet.
	 **/
	typedef boost::shared_ptr< AbstractDataSet > ADataSetPtr;

	const M4D::Common::TimeStamp&
	GetStructureTimestamp()const
		{ return _structureTimestamp; }
protected:
	/**
	 * Increase structure timestamp - only helper function for successors.
	 **/
	void
	IncStructureTimestamp()
		{ ++_structureTimestamp; }

	/**
	 * Time stamp of structure changes. When internal structure is 
	 * changed (reallocation of buffers, etc.) timestamp is increased.
	 **/
	M4D::Common::TimeStamp	_structureTimestamp;
private:

};


}/*namespace Imaging*/
}/*namespace M4D*/

#endif /*_ABSTRACT_DATA_SET_H*/

