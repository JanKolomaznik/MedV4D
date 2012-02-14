#ifndef MOUSE_TRACKING_H
#define MOUSE_TRACKING_H

#include "MedV4D/Common/Vector.h"

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

struct Mouse3DTrackInfo: public MouseTrackInfo
{
	void
	startTracking( QPoint aLocalPosition, QPoint aGlobalPosition, Vector3f aSpaceCoords ) 
	{
		MouseTrackInfo::startTracking( aLocalPosition, aGlobalPosition );
		mStartSpaceCoords = mLastSpaceCoords = aSpaceCoords;
	}

	QPoint
	trackUpdate( QPoint aLocalPosition, QPoint aGlobalPosition, Vector3f aSpaceCoords, Vector3f &aSpaceDiff )
	{
		QPoint diff = MouseTrackInfo::trackUpdate( aLocalPosition, aGlobalPosition );
		aSpaceDiff = aSpaceCoords - mLastSpaceCoords;
		mLastSpaceCoords = aSpaceCoords;
		return diff;
	}
	
	Vector3f mStartSpaceCoords;
	Vector3f mLastSpaceCoords;
};

} /*namespace Viewer*/
} /*namespace GUI*/
} /*namespace M4D*/

#endif /*MOUSE_TRACKING_H*/
