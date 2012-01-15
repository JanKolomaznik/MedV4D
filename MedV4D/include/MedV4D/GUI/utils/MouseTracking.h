#ifndef MOUSE_TRACKING_H
#define MOUSE_TRACKING_H

namespace M4D
{
namespace GUI
{
namespace Viewer
{
	


struct MouseTrackInfo
{
	void
	startTracking( QPoint aLocalPosition, QPoint aGlobalPosition ) 
	{
		mStartLocalPosition = mLastLocalPosition = aLocalPosition;

		mStartGlobalPosition = mLastGlobalPosition = aGlobalPosition;
	}

	QPoint
	trackUpdate( QPoint aLocalPosition, QPoint aGlobalPosition )
	{
		QPoint diff = aLocalPosition - mLastLocalPosition;
		mLastLocalPosition = aLocalPosition;

		mLastGlobalPosition = aGlobalPosition;
		return diff;
	}

	QPoint	mStartLocalPosition;
	QPoint	mLastLocalPosition;

	QPoint	mStartGlobalPosition;
	QPoint	mLastGlobalPosition;
};

} /*namespace Viewer*/
} /*namespace GUI*/
} /*namespace M4D*/

#endif /*MOUSE_TRACKING_H*/
