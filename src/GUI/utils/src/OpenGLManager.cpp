#include "GUI/utils/OpenGLManager.h"

struct TextureRecord
	{
		M4D::GLTextureImage::Ptr texture;
		M4D::Common::TimeStamp sourceTimeStamp; //<Contents timestamp
	};
typedef std::map< M4D::Common::TimeStamp::IDType, TextureRecord > TextureStorage;

struct OpenGLManagerPimpl
{
	QGLWidget *widget;
	QGLContext *context;
	
	TextureStorage textureStorage; //<Indexed by ID from structural timestamp
	mutable boost::recursive_mutex mTextureMutex;
};

OpenGLManager *oglManagerInstance = NULL;
//*******************************************************************************************
OpenGLManager *
OpenGLManager::getInstance()
{
	ASSERT( oglManagerInstance );
	return oglManagerInstance;
}

OpenGLManager::OpenGLManager( OpenGLManager *aInstance )
{
	ASSERT( aInstance );
	oglManagerInstance = aInstance;
}

OpenGLManager::~OpenGLManager()
{

}

void
OpenGLManager::initialize()
{
	mPimpl = new OpenGLManagerPimpl;

	mPimpl->widget = new QGLWidget();
	mPimpl->context = const_cast< QGLContext * >( mPimpl->widget->context() );
}

void
OpenGLManager::finalize()
{
	makeCurrent();
	mPimpl->textureStorage.clear();
	doneCurrent();

	delete mPimpl->widget;

	delete mPimpl;
}

QGLWidget *
OpenGLManager::getSharedGLWidget()
{
	ASSERT( mPimpl );
	return mPimpl->widget;
}

M4D::GLTextureImage::Ptr
OpenGLManager::getTextureFromImage( const M4D::Imaging::AImage &aImage )
{
	ASSERT( mPimpl );
	boost::unique_lock< boost::recursive_mutex > lock( mPimpl->mTextureMutex );

	
	M4D::Common::TimeStamp structTimestamp( aImage.GetStructureTimestamp() );
	M4D::Common::TimeStamp::IDType id = structTimestamp.getID();

	M4D::Common::TimeStamp timestamp( aImage.GetEditTimestamp() );

	TextureStorage::iterator it = mPimpl->textureStorage.find( id );
	if ( it != mPimpl->textureStorage.end() ) { //We already created one instance
		if ( timestamp == it->second.sourceTimeStamp ) {
			D_PRINT( "Returning valid instance" );
			return it->second.texture;
		} 
		//it->second.texture->DeleteTexture();
	}

	makeCurrent();
	TextureRecord rec;
	rec.texture = M4D::CreateTextureFromImage( *(aImage.GetAImageRegion()), true ) ;
	rec.sourceTimeStamp = timestamp;
	mPimpl->textureStorage[ id ] = rec;
	doneCurrent();	
	return rec.texture;		
}

void
OpenGLManager::makeCurrent()
{
	//TODO handle in better way
	mPimpl->context->makeCurrent();
}

void
OpenGLManager::doneCurrent()
{
	//TODO handle in better way
	mPimpl->context->doneCurrent();
}


