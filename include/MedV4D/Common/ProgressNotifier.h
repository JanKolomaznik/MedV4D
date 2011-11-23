#ifndef PROGRESS_NOTIFIER_H
#define PROGRESS_NOTIFIER_H


#include <boost/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/shared_ptr.hpp>
#include <string>

#include "MedV4D/Common/Debug.h"
#include "MedV4D/Common/Log.h"
#include "MedV4D/Common/StringTools.h"

class ProgressNotifier
{
public:
	typedef boost::shared_ptr< ProgressNotifier > Ptr;

	virtual void
	init( size_t aPhaseCount )
	{
		ASSERT( aPhaseCount > 0 );
		boost::mutex::scoped_lock( mAccessMutex );
		mPhaseCount = aPhaseCount;
		mCanceled = false;
	}

	virtual void
	initNextPhase( size_t aStepCount )
	{
		boost::mutex::scoped_lock( mAccessMutex );
		++mCurrentPhase;
		mStepCount = aStepCount;
		mCurrentStep = 0;
	}

	virtual bool
	updateProgress( int aStepProgress = 1 )
	{
		return updateProgressThreadSafe( aStepProgress );
	}

	/*virtual void
	start()
	{ 
		mCurrentStep = 0;
		mCanceled = false;
	}*/

	virtual void
	finished()
	{ }

	virtual void
	cancel()
	{ 
		boost::mutex::scoped_lock( mAccessMutex );
		mCanceled = true;
	}

	bool
	isCanceled()
	{
		return mCanceled;
	}

	virtual void
	pause()
	{ }
protected:
	void
	setInfoTextThreadSafe( const std::wstring &aInfoText )
	{
		boost::mutex::scoped_lock( mAccessMutex );
		mInfoText = aInfoText;
	}
	void
	setStepCountThreadSafe( size_t aCount )
	{
		boost::mutex::scoped_lock( mAccessMutex );
		mStepCount = aCount;
	}

	bool
	updateProgressThreadSafe( int aStepProgress )
	{
		//TODO assert
		boost::mutex::scoped_lock( mAccessMutex );
		mCurrentStep += aStepProgress;

		return mCanceled;
	}

	

	std::wstring	mInfoText;
	size_t		mStepCount;
	size_t		mCurrentStep;

	size_t		mPhaseCount;
	size_t		mCurrentPhase;

	
	mutable boost::mutex mAccessMutex;
	bool		mCanceled;


};

/*class ProgressNotifier
{
public:
	virtual void
	setInfoText( const std::wstring &aInfoText )
	{
		setInfoTextThreadSafe( aInfoText );
	}

	virtual void
	setStepCount( size_t aCount )
	{
		setStepCountThreadSafe( aCount );
	}

	virtual void
	updateProgress( int aStepProgress = 1 )
	{
		updateProgressThreadSafe( aStepProgress );
	}

	virtual void
	start()
	{ 
		mCurrentStep = 0;
		mCanceled = false;
	}

	virtual void
	finished()
	{ }

	virtual void
	cancel()
	{ 
		boost::mutex::scoped_lock( mAccessMutex );
		mCanceled = true;
	}

	bool
	isCanceled()
	{
		return mCanceled;
	}

	virtual void
	pause()
	{ }
protected:
	void
	setInfoTextThreadSafe( const std::wstring &aInfoText )
	{
		boost::mutex::scoped_lock( mAccessMutex );
		mInfoText = aInfoText;
	}
	void
	setStepCountThreadSafe( size_t aCount )
	{
		boost::mutex::scoped_lock( mAccessMutex );
		mStepCount = aCount;
	}

	void
	updateProgressThreadSafe( int aStepProgress )
	{
		//TODO assert
		boost::mutex::scoped_lock( mAccessMutex );
		mCurrentStep += aStepProgress;
	}

	

	std::wstring	mInfoText;
	size_t		mStepCount;
	size_t		mCurrentStep;

	
	mutable boost::mutex mAccessMutex;
	bool		mCanceled;


};*/

#endif /*PROGRESS_NOTIFIER_H*/
