#include "MedV4D/GUI/utils/TransferFunctionBuffer.h"

#include "MedV4D/GUI/utils/OGLDrawing.h"

namespace M4D
{
namespace GUI
{

TransferFunctionBuffer1D::TransferFunctionBuffer1D( size_t aSize, MappedInterval aMappedInterval )
	: mBuffer( NULL ), mSize( aSize ), mMappedInterval( aMappedInterval )
{
	Resize( aSize );
}

TransferFunctionBuffer1D::~TransferFunctionBuffer1D()
{
	if ( mBuffer != NULL ) {
		delete mBuffer;
	}
}

TransferFunctionBuffer1D::Iterator
TransferFunctionBuffer1D::Begin()
{
	ASSERT( mBuffer != NULL || mSize == 0 );
	return mBuffer;	
}

TransferFunctionBuffer1D::Iterator
TransferFunctionBuffer1D::End()
{
	ASSERT( mBuffer != NULL || mSize == 0 );
	return mBuffer + mSize;
}

TransferFunctionBuffer1D::ConstIterator
TransferFunctionBuffer1D::Begin()const
{
	ASSERT( mBuffer != NULL || mSize == 0 );
	return mBuffer;	
}

TransferFunctionBuffer1D::ConstIterator
TransferFunctionBuffer1D::End()const
{
	ASSERT( mBuffer != NULL || mSize == 0 );
	return mBuffer + mSize;
}

TransferFunctionBuffer1D::Iterator
TransferFunctionBuffer1D::GetNearest( float32 aValue )
{
	int idx = round( ( aValue - mMappedInterval[0] ) / ( mMappedInterval[1] - mMappedInterval[0] ) * (float)mSize );
	if ( idx < 0 || idx >= (int)mSize ) {
		return End();
	}
	
	return &(mBuffer[ idx ]);
}

TransferFunctionBuffer1D::ConstIterator
TransferFunctionBuffer1D::GetNearest( float32 aValue )const
{
	int idx = round( ( aValue - mMappedInterval[0] ) / ( mMappedInterval[1] - mMappedInterval[0] ) * (float)mSize );
	if ( idx < 0 || idx >= (int)mSize ) {
		return End();
	}
	
	return &(mBuffer[ idx ]);
}

int
TransferFunctionBuffer1D::GetNearestIndex( float32 aValue )const
{
	int idx = round( ( aValue - mMappedInterval[0] ) / ( mMappedInterval[1] - mMappedInterval[0] ) * (float)mSize );
	if ( idx < 0 ) {
		return -1;
	}
	if( idx >= (int)mSize ) {
		return static_cast<int>(mSize);
	}
	return idx;
}

size_t
TransferFunctionBuffer1D::Size()const
{
	ASSERT( mBuffer != NULL || mSize == 0 );
	return mSize;
}

void
TransferFunctionBuffer1D::Resize( size_t aSize )
{
	if ( mBuffer != NULL ) {
		mSize = 0;
		delete mBuffer;
	}

	if ( aSize == 0 ) {
		mBuffer = NULL;
	}
	mSize = 0;

	mBuffer = new ValueType[ aSize ];
	if ( mBuffer == NULL ) {
		_THROW_ ErrorHandling::EAllocationFailed();
	}
	mSize = aSize;
}

TransferFunctionBuffer1D::MappedInterval
TransferFunctionBuffer1D::GetMappedInterval()const
{
	return mMappedInterval;
}

void
TransferFunctionBuffer1D::SetMappedInterval( TransferFunctionBuffer1D::MappedInterval aMappedInterval )
{
	mMappedInterval = aMappedInterval;
}



GLTransferFunctionBuffer1D::Ptr
CreateGLTransferFunctionBuffer1D( const TransferFunctionBuffer1D &aTransferFunction )
{
	if ( aTransferFunction.Size() == 0 ) {
		_THROW_ ErrorHandling::EBadParameter( "Transfer function buffer of 0 size" );
	}

	GLuint texName;

	try {
		GL_CHECKED_CALL( glPixelStorei( GL_UNPACK_ALIGNMENT, 1 ) );
		GL_CHECKED_CALL( glPixelStorei(GL_PACK_ALIGNMENT, 1) );
		GL_CHECKED_CALL( glGenTextures( 1, &texName ) );
		GL_CHECKED_CALL( glBindTexture ( GL_TEXTURE_1D, texName ) );
		GL_CHECKED_CALL( glTexEnvf( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE ) );

		GL_CHECKED_CALL( glTexParameteri( GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE ) );
		GL_CHECKED_CALL( glTexParameteri( GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR ) );
		GL_CHECKED_CALL( glTexParameteri( GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR ) );


		GL_CHECKED_CALL( glEnable( GL_TEXTURE_1D ) );
		
		GL_CHECKED_CALL( glBindTexture( GL_TEXTURE_1D, texName ) );

		GL_CHECKED_CALL( 
			glTexImage1D(
				GL_TEXTURE_1D, 
				0, 
				GL_RGBA32F, 
				static_cast<GLsizei>(aTransferFunction.Size()), 
				0, 
				GL_RGBA, 
				GL_FLOAT, 
				aTransferFunction.Begin()
				)
			);

		
		M4D::CheckForGLError( "OGL building texture for transfer function: " );
	} 
	catch( ... ) {
		if( texName != 0 ) {
			glDeleteTextures( 1, &texName );
		}
		throw;
	}

	return GLTransferFunctionBuffer1D::Ptr( new GLTransferFunctionBuffer1D( texName, aTransferFunction.GetMappedInterval(), aTransferFunction.Size() ) );

}

} /*namespace GUI*/
} /*namespace M4D*/

