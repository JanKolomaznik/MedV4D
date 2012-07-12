#ifndef ID_MAPPING_BUFFER_H
#define ID_MAPPING_BUFFER_H

#include "MedV4D/Common/Common.h"
#include "MedV4D/Common/DefinitionMacros.h"
#include "MedV4D/GUI/utils/OGLTools.h"

#include <vector>

namespace M4D
{
namespace GUI
{
	
class IDMappingBuffer
{
public:
	SMART_POINTER_TYPEDEFS( IDMappingBuffer );

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

struct GLIDMappingBuffer
{
	SMART_POINTER_TYPEDEFS( GLIDMappingBuffer );
	/*typedef Vector2f MappedInterval;
	typedef boost::shared_ptr< GLTransferFunctionBuffer1D > Ptr;
	typedef boost::shared_ptr< const GLTransferFunctionBuffer1D > ConstPtr;
	typedef boost::weak_ptr< GLTransferFunctionBuffer1D > WPtr;
	typedef boost::weak_ptr< const GLTransferFunctionBuffer1D > ConstWPtr;

	friend GLTransferFunctionBuffer1D::Ptr createGLTransferFunctionBuffer1D( const TransferFunctionBuffer1D &aTransferFunction );

	MappedInterval
	getMappedInterval()const
	{ return mMappedInterval; }

	GLuint
	getTextureID()const
	{ return mGLTextureID; }

	int
	getSampleCount()const
	{
		return mSampleCount;
	}*/
private:
	/*GLTransferFunctionBuffer1D( GLuint aGLTextureID, MappedInterval aMappedInterval, int aSampleCount )
		: mGLTextureID( aGLTextureID ), mMappedInterval( aMappedInterval ), mSampleCount( aSampleCount )
	{ }

	GLuint	mGLTextureID;
	MappedInterval mMappedInterval;
	int mSampleCount;*/
};

GLIDMappingBuffer::Ptr
createGLMappingBuffer( const IDMappingBuffer &aBuffer );

struct IDMappingBufferInfo
{
	IDMappingBufferInfo():id(0)
	{ }

	M4D::Common::IDNumber id;
	M4D::GUI::GLIDMappingBuffer::Ptr glBuffer;
	M4D::GUI::IDMappingBuffer::Ptr buffer;
};


} /*namespace GUI*/
} /*namespace M4D*/


#endif //ID_MAPPING_BUFFER_H
