#ifndef TRANSFER_FUNCTION_BUFFER_H
#define TRANSFER_FUNCTION_BUFFER_H

#include "MedV4D/Common/Common.h"
#include "MedV4D/GUI/utils/OGLTools.h"

namespace M4D
{
namespace GUI
{
	
class TransferFunctionBuffer1D
{
public:
	typedef boost::shared_ptr< TransferFunctionBuffer1D > Ptr;

	typedef RGBAf 			ValueType;
	typedef ValueType *		Iterator;
	typedef const ValueType*	ConstIterator;

	typedef ValueType 		value_type;
	typedef Iterator 		iterator;
	typedef ConstIterator 		const_iterator;

	typedef Vector2f		MappedInterval;

	TransferFunctionBuffer1D( size_t aSize = 0, MappedInterval aMappedInterval = MappedInterval( 0.0f, 1.0f ) );

	~TransferFunctionBuffer1D();


	Iterator
	Begin();

	Iterator
	End();

	ConstIterator
	Begin()const;

	ConstIterator
	End()const;

	size_t
	Size()const;

	void
	Resize( size_t aSize );

	MappedInterval
	GetMappedInterval()const;

	Iterator
	GetNearest( float32 aValue );
	
	ConstIterator
	GetNearest( float32 aValue )const;

	int
	GetNearestIndex( float32 aValue )const;

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

	void
	SetMappedInterval( MappedInterval aMappedInterval );
protected:
	RGBAf	*mBuffer;
	size_t	mSize;
	MappedInterval mMappedInterval;
private:

};

struct GLTransferFunctionBuffer1D;

struct GLTransferFunctionBuffer1D
{
	typedef Vector2f MappedInterval;
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
	}
private:
	GLTransferFunctionBuffer1D( GLuint aGLTextureID, MappedInterval aMappedInterval, int aSampleCount )
		: mGLTextureID( aGLTextureID ), mMappedInterval( aMappedInterval ), mSampleCount( aSampleCount )
	{ /* empty */ }

	GLuint	mGLTextureID;
	MappedInterval mMappedInterval;
	int mSampleCount;
};

GLTransferFunctionBuffer1D::Ptr
createGLTransferFunctionBuffer1D( const TransferFunctionBuffer1D &aTransferFunction );

struct TransferFunctionBufferInfo
{
	/*TransferFunctionBufferInfo( M4D::Common::IDNumber aId, M4D::GUI::GLTransferFunctionBuffer1D::Ptr aTfGLBuffer, M4D::GUI::TransferFunctionBuffer1D::Ptr aTfBuffer ):
		id(aId), tfGLBuffer( aTfGLBuffer ), tfBuffer( aTfBuffer )
	{ }*/
	TransferFunctionBufferInfo():id(0)
	{ }

	M4D::Common::IDNumber id;
	M4D::GUI::GLTransferFunctionBuffer1D::Ptr tfGLBuffer;
	M4D::GUI::TransferFunctionBuffer1D::Ptr tfBuffer;

	M4D::GUI::GLTransferFunctionBuffer1D::Ptr tfGLIntegralBuffer;
	M4D::GUI::TransferFunctionBuffer1D::Ptr tfIntegralBuffer;
};


} /*namespace GUI*/
} /*namespace M4D*/

#endif /*TRANSFER_FUNCTION_BUFFER_H*/
