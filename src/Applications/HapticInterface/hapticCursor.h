#ifndef HAPTIC_VIEWER_HAPTIC_CURSOR
#define HAPTIC_VIEWER_HAPTIC_CURSOR

#include "cursorInterface.h"
#define _MSVC
#include <chai3d.h>
#include "common/Log.h"

namespace M4D
{
	namespace Viewer
	{
		class hapticCursor : public cursorInterface
		{
		public:
			hapticCursor(Imaging::InputPortTyped< Imaging::AImage >	*inPort) : cursorInterface(inPort){};
			void startHaptics();
		private:
			void updateHaptics();
			bool runHpatics;
			// a haptic device handler
			cHapticDeviceHandler* handler;
			// a pointer to a haptic device
			cGenericHapticDevice* hapticDevice;
			// haptic device info
			cHapticDeviceInfo info;
			// number of haptic devices
			int numHapticDevices;
			// last position of haptic device
			cVector3d position;
			// haptics thread
			cThread* hapticsThread;

			int _sizeX, _sizeY, _sizeZ;
			int _minX, _minY, _minZ;
			int maxSize;
			int64 _minValue, _maxValue;
			int _imageID;
		};
	}
}

#endif