#ifndef TRANSFER_FUNCTION_BUFFER_H
#define TRANSFER_FUNCTION_BUFFER_H

#include "common/Common.h"
#include "common/OGLTools.h"

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

	void
	SetMappedInterval( MappedInterval aMappedInterval );
protected:
	RGBAf	*mBuffer;
	size_t	mSize;
	MappedInterval mMappedInterval;
private:

};

struct GLTransferFunctionBuffer1D;

GLTransferFunctionBuffer1D
CreateGLTransferFunctionBuffer1D( const TransferFunctionBuffer1D &aTransferFunction );

struct GLTransferFunctionBuffer1D
{
	friend GLTransferFunctionBuffer1D CreateGLTransferFunctionBuffer1D( const TransferFunctionBuffer1D &aTransferFunction );
	typedef Vector2f MappedInterval;


	MappedInterval
	GetMappedInterval()const
	{ return mMappedInterval; }

	GLuint
	GetTextureID()const
	{ return mGLTextureID; }
private:
	GLTransferFunctionBuffer1D( GLuint aGLTextureID, MappedInterval aMappedInterval )
		: mGLTextureID( aGLTextureID ), mMappedInterval( aMappedInterval )
	{ /* empty */ }

	GLuint	mGLTextureID;
	MappedInterval mMappedInterval;
};

} /*namespace GUI*/
} /*namespace M4D*/

#endif /*TRANSFER_FUNCTION_BUFFER_H*/
