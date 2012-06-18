#ifndef RAII_H
#define RAII_H

#include <boost/function.hpp>
#include <boost/utility.hpp>


namespace M4D
{


class RAII 
{
public:
	template< typename TAcquisition, typename TRelease >
	RAII( TAcquisition aAcquisition, TRelease aRelease, bool aAcquire = true ): mAcquisition( aAcquisition ), mRelease( aRelease ), mAcquired( false )
	{ 
		if ( aAcquire ) {
			acquire();
		}
		
	}
	
	~RAII()
	{
		release();
	}
	
	void
	acquire()
	{
		if( ! mAcquired ) {
			mAcquisition();
			mAcquired = true;
		}
	}
	
	void
	release()
	{
		if( mAcquired ) {
			mRelease();
			mAcquired = false;
		}
	}
	
protected:
	boost::function< void() > mAcquisition;
	boost::function< void() > mRelease;
	bool mAcquired;
};

template< typename TResource >
class ResourceGuard
{
public:
	template< typename TAcquisition, typename TRelease >
	ResourceGuard( TAcquisition aAcquisition, TRelease aRelease, bool aAcquire = true ): mAcquisition( aAcquisition ), mRelease( aRelease ), mAcquired( false ), mValid( true )
	{ 
		if ( aAcquire ) {
			acquire();
		}
		
	}

	ResourceGuard(): mAcquired( false ), mValid( false )
	{}
	
	~ResourceGuard()
	{
		release();
	}
	
	void
	acquire()
	{
		if( !mValid ) {
			_THROW_ ErrorHandling::EObjectUnavailable( "acquire() failed. Resource guard not valid" );
		}
		if( ! mAcquired ) {
			mResource = mAcquisition();
			mAcquired = true;
		}
	}
	
	void
	release()
	{
		
		if( mAcquired ) {
			/*if( !mValid ) {
				_THROW_ ErrorHandling::EObjectUnavailable( "Resource guard not valid" );
			}*/
			mRelease( mResource );
			mAcquired = false;
		}
	}
	TResource &
	get()
	{ 
		if( !mValid ) {
			_THROW_ ErrorHandling::EObjectUnavailable( "get() failed. Resource guard not valid" );
		}
		if( !mAcquired ) {
			_THROW_ ErrorHandling::EObjectUnavailable( "Resource not acquired!" );
		}
		return mResource; 
	}
protected:

	boost::function< TResource() > mAcquisition;
	boost::function< void(TResource &) > mRelease;
	bool mAcquired;
	bool mValid;

	TResource mResource;

};

} //M4D

#endif /*RAII_H*/
