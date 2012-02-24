#ifndef RAII_H
#define RAII_H

#include <boost/function.hpp>


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

} //M4D

#endif /*RAII_H*/