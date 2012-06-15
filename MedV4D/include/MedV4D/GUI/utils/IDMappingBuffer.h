#ifndef ID_MAPPING_BUFFER_H
#define ID_MAPPING_BUFFER_H

#include "MedV4D/Common/Common.h"
#include "MedV4D/GUI/utils/OGLTools.h"

#include <vector>

namespace M4D
{
namespace GUI
{
	
class IDMappingBuffer
{
public:
	typedef boost::shared_ptr< IDMappingBuffer > Ptr;

	typedef RGBAf 				ValueType;
	typedef std::vector< ValueType >		InternalBuffer;
	typedef InternalBuffer::iterator		Iterator;
	typedef InternalBuffer::const_iterator	ConstIterator;

	typedef ValueType 		value_type;
	typedef Iterator 		iterator;
	typedef ConstIterator 		const_iterator;

	IDMappingBuffer( size_t aSize = 0 ): mBuffer( aSize )
	{ /*empty*/ }

	
	Iterator
	begin()
	{ return mBuffer.begin(); }

	Iterator
	end()
	{ return mBuffer.end(); }

	ConstIterator
	begin()const
	{ return mBuffer.begin(); }

	ConstIterator
	end()const
	{ return mBuffer.end(); }

	size_t
	size()const
	{ return mBuffer.size(); }

	void
	resize( size_t aSize )
	{ mBuffer.resize( aSize ); }

	ValueType &
	operator[]( size_t aIdx )
	{
		ASSERT( aIdx < mSize );
		return mBuffer[ aIdx ];
	}

	ValueType
	operator[]( size_t aIdx ) const
	{
		ASSERT( aIdx < mSize );
		return mBuffer[ aIdx ];
	}
protected:
	std::vector< RGBAf > mBuffer;
	size_t	mSize;
private:

};


#endif //ID_MAPPING_BUFFER_H