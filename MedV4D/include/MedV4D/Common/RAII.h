#ifndef RAII_H
#define RAII_H

#include <functional>
#include <boost/utility.hpp>
#include <memory>



namespace M4D
{


class RAII : private boost::noncopyable
{
public:
	template< typename TAcquisition, typename TRelease >
	RAII( TAcquisition aAcquisition, TRelease aRelease, bool aAcquire = true )
		: mAcquisition( aAcquisition )
		, mRelease( aRelease )
		, mAcquired( false )
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
	std::function< void() > mAcquisition;
	std::function< void() > mRelease;
	bool mAcquired;
};

template< typename TResource >
class ResourceGuard : private boost::noncopyable
{
public:
	ResourceGuard( std::function< TResource() > aAcquisition, std::function< void(TResource &) > aRelease, bool aAcquire = true )
		: mAcquisition( aAcquisition )
		, mRelease( aRelease )
		, mAcquired( false )
		, mValid( true )
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
			D_PRINT( "Acquiring resource" );
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
			D_PRINT( "Releasing resource" );
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

	std::function< TResource() > mAcquisition;
	std::function< void(TResource &) > mRelease;
	bool mAcquired;
	bool mValid;

	TResource mResource;

};

template< typename TResource >
std::shared_ptr< ResourceGuard< TResource > >
makeResourceGuardPtr(std::function< TResource() > aAcquisition, std::function< void(TResource &) > aRelease, bool aAcquire = true)
{
	return std::shared_ptr< ResourceGuard< TResource > >( new ResourceGuard< TResource >( aAcquisition, aRelease, aAcquire ) );
}

} //M4D

#endif /*RAII_H*/
