#ifndef M4D_GUI_OPENGLHAPTICVIEWERWIDGET_H_HAPTIC_VIEWER_HAPTIC_CURSOR
#define M4D_GUI_OPENGLHAPTICVIEWERWIDGET_H_HAPTIC_VIEWER_HAPTIC_CURSOR

#include "cursorInterface.h"
#define _MSVC
#include <chai3d.h>
#include "common/Log.h"
#include "transitionFunction.h"

namespace M4D
{
	namespace Viewer
	{
		class hapticCursor : public cursorInterface
		{
		public:
			hapticCursor(vtkImageData* input);
			~hapticCursor();
			void startHaptics();
			void stop();
		protected:

			class hapticDeviceWorker
			{
			public:
				hapticDeviceWorker(cGenericHapticDevice* hapticDevice, hapticCursor* supervisor, bool* runHaptic);
				void operator()();
			protected:
				cGenericHapticDevice* hapticDevice; // pointer to haptic device that this class communicate with
				hapticCursor* supervisor; // class of haptic cursor where to pass position
				bool* runHaptic; // indicates if continue to listen or not
			};
			virtual void StartListen(); // method which starts new thread where haptics is running
			virtual void SetCursorPosition(const cVector3d& cursorPosition); // Main method which sets cursor position and counts force for that position
			cVector3d& GetForce();
			bool runHpatics; // indicates if continue to listen or not
			cHapticDeviceHandler* handler; // a haptic device handler
			cGenericHapticDevice* hapticDevice; // a pointer to a haptic device
			cHapticDeviceInfo info; // haptic device infos
			int numHapticDevices; // number of haptic devices
			hapticDeviceWorker* deviceWorker; // class which listen to haptic in fact
			cVector3d force; // force to set to haptic
			boost::thread* hapticsThread; // pointer to thread where haptics is running;
			transitionFunction* hapticForceTransitionFunction; // transition function for setting force for haptic device
		};
	}
}

#endif