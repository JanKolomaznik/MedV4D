#ifndef M4D_GUI_OPENGLHAPTICVIEWERWIDGET_H_HAPTIC_VIEWER_HAPTIC_CURSOR
#define M4D_GUI_OPENGLHAPTICVIEWERWIDGET_H_HAPTIC_VIEWER_HAPTIC_CURSOR

#include "cursorInterface.h"
#define _MSVC
#include <chai3d.h>
#include "common/Log.h"
#include "transitionFunction.h"
#include "vtkRenderWindow.h"
#include <vector>
#include <iostream>

namespace M4D
{
	namespace Viewer
	{
		class hapticCursor : public cursorInterface
		{
		public:
			hapticCursor(vtkImageData* input, vtkRenderWindow* renderWindow, transitionFunction* hapticForceTransitionFunction);
			hapticCursor(vtkRenderWindow* renderWindow, transitionFunction* hapticForceTransitionFunction);
			~hapticCursor();
			void startHaptics();
			void stop();
			virtual int GetValue(); // returns value of point where the cursor stands
			virtual void SetTraceLogOn( std::string file );
			virtual void SetTraceLogOff();
		protected:
			virtual void StartListen(); // method which starts new thread where haptics is running
			virtual void SetCursorPosition(const cVector3d& cursorPosition); // Main method which sets cursor position and counts force for that position
			virtual void SetZoomInButtonPressed(bool pressed); // set button pressed status
			virtual void SetZoomOutButtonPressed(bool pressed); // set button pressed status
			virtual void deviecWorker();
			cVector3d& GetForce();
			cVector3d lastPosition;
			bool runHaptics; // indicates if continue to listen or not
			cHapticDeviceHandler* handler; // a haptic device handler
			cGenericHapticDevice* hapticDevice; // a pointer to a haptic device
			vtkRenderWindow* renderWindow;
			cHapticDeviceInfo info; // haptic device infos
			int numHapticDevices; // number of haptic devices
			cVector3d force; // force to set to haptic
			boost::thread* hapticsThread; // pointer to thread where haptics is running;
			transitionFunction* hapticForceTransitionFunction; // transition function for setting force for haptic device
			bool zoomInButtonPressed, zoomOutButtonPressed; // Indication whether buttons on haptic device are pushed or not
			cPrecisionClock* m_clock;
			int64 count;
			boost::mutex runMutex;
			double epsilon, ksi;
			double springPower;
			int value;
			std::vector< cVector3d > vectors;
			int numberOfVectors;
			bool proxyMode;
			cVector3d solidPlaneParams;
			double dParamOfPlane;
			std::ofstream traceLogFile;
			bool traceLogEnabled;
		};
	}
}

#endif