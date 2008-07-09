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

	/*template< typename DatasetType >
	static AbstractDataSet &
	CastDataSet(  DatasetType & dataset )
	{
		//TODO - handle exception well
		return dynamic_cast< AbstractDataSet & >( dataset );
	}

	template< typename DatasetType >
	static const AbstractDataSet &
	CastDataSet( const DatasetType & dataset )
	{
		//TODO - handle exception well
		return dynamic_cast< const AbstractDataSet & >( dataset );
	}*/

	template< typename DatasetType >
	static typename AbstractDataSet::ADataSetPtr
	CastDataSet( boost::shared_ptr< DatasetType > & dataset )
	{
		if( dynamic_cast< DatasetType * >( dataset.get() ) == NULL ) {
		//TODO throw exception
		}

		return boost::static_pointer_cast< AbstractDataSet >( dataset );
	}
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

