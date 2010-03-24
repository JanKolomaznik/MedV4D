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
			hapticCursor(Imaging::InputPortTyped< Imaging::AImage >	*inPort);
			~hapticCursor();
			void startHaptics();
			void stop();
		protected:

			class hapticDeviceWorker
			{
			public:
				hapticDeviceWorker(cGenericHapticDevice* hapticDevice, hapticCursor* supervisor, bool* runHaptic);
				void StartListen();
				void operator()();
			protected:
				cGenericHapticDevice* hapticDevice; // pointer to haptic device that this class communicate with
				hapticCursor* supervisor; // class of haptic cursor where to pass position
				bool* runHaptic; // indicates if continue to listen or not
			};

			virtual void SetCursorPosition(cVector3d& cursorPosition)
			{
				// TODO musi se spocitat pozice kurzoru vuci scale a pozici krychle !! zatim spatne
				cVector3d lastPosition = this->cursorPosition;
				this->cursorPosition = cursorPosition;
				cVector3d difference = cursorPosition - lastPosition;
				difference.normalize();
				// TODO dopocitat vektor sily - prechodova Fce a nastavit ho do this->force !!
			}
			cVector3d& GetForce();
			bool runHpatics; // indicates if continue to listen or not
			cHapticDeviceHandler* handler; // a haptic device handler
			cGenericHapticDevice* hapticDevice; // a pointer to a haptic device
			cHapticDeviceInfo info; // haptic device infos
			int numHapticDevices; // number of haptic devices
			hapticDeviceWorker* deviceWorker; // class which listen to haptic in fact
			cVector3d force; // force to set to haptic
		};
	}
}

#endif