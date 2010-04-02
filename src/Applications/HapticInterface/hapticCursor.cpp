#include "hapticCursor.h"

namespace M4D
{
	namespace Viewer
	{
#pragma region hapticDeviceWorker
		hapticCursor::hapticDeviceWorker::hapticDeviceWorker(cGenericHapticDevice* hapticDevice, hapticCursor* supervisor, bool* runHaptic)
		{
			this->hapticDevice = hapticDevice;
			this->supervisor = supervisor;
			this->runHaptic = runHaptic;
		}
		void hapticCursor::hapticDeviceWorker::operator()()
		{
			while(*runHaptic)
			{
				cVector3d hapticPosition;
				hapticDevice->getPosition(hapticPosition);
				supervisor->SetCursorPosition(hapticPosition);
				hapticDevice->setForce(supervisor->GetForce());
			}
		}
#pragma endregion hapticDeviceWorker

		hapticCursor::hapticCursor(vtkImageData *input) : cursorInterface(input)
		{
			handler = new cHapticDeviceHandler();
			deviceWorker = NULL;
			hapticDevice = NULL;
			runHpatics = false;
			hapticForceTransitionFunction = new transitionFunction(minVolumeValue, maxVolumeValue, 0.0, 1.0);
		}
		hapticCursor::~hapticCursor()
		{
			stop();
			delete(handler);
			if (hapticDevice != NULL)
				delete(hapticDevice);
			if (deviceWorker != NULL)
				delete(deviceWorker);
			delete(hapticForceTransitionFunction);
		}
		void hapticCursor::startHaptics()
		{
			numHapticDevices = handler->getNumDevices();
			if (numHapticDevices > 0)
			{
				handler->getDevice(hapticDevice, 0); // get haptic device from handler
				hapticDevice->open(); // open connection to haptic device
				hapticDevice->initialize(); // initialize haptic device	
				info = hapticDevice->getSpecifications(); // retrieve information about the current haptic device
				std::cout << "Chai3d: " << info.m_manufacturerName << " " << info.m_modelName << std::endl;
				deviceWorker = new hapticDeviceWorker(hapticDevice, this, &runHpatics);
				StartListen();
			}
			else
			{
				std::cout << "No haptic device found!" << std::endl;
			}
		}
		void hapticCursor::SetCursorPosition(const cVector3d& cursorPosition)
		{
			double center[3];
			cursor->GetCenter(center);
			cVector3d lastRealPosition(center[0], center[1], center[2]);

			double radiusCubeCenter[3];
			cursorRadiusCube->GetCenter(radiusCubeCenter);
			cVector3d radiusCubeCenterChai(radiusCubeCenter[0], radiusCubeCenter[1], radiusCubeCenter[2]);
			cVector3d newRealPosition = radiusCubeCenterChai + cursorPosition / info.m_workspaceRadius * scale / 2; // radius gets just half of length of cube edge but scale means full length
			double newRealPositionVTK[3];
			newRealPositionVTK[0] = newRealPosition.x;
			newRealPositionVTK[1] = newRealPosition.y;
			newRealPositionVTK[2] = newRealPosition.z;
			cursor->SetCenter(newRealPositionVTK);

			cVector3d newForce = newRealPosition - lastRealPosition;
			newForce.normalize();
			unsigned short cursorVolumeValue = (unsigned short)input->GetScalarComponentAsDouble((int)newRealPositionVTK[0], (int)newRealPositionVTK[1], (int)newRealPositionVTK[2], 0);
			newForce *= hapticForceTransitionFunction->GetValueOnPoint(cursorVolumeValue);

			force = newForce;
		}
		cVector3d& hapticCursor::GetForce()
		{
			return force;
		}
		void hapticCursor::stop()
		{
			if (runHpatics)
			{
				runHpatics = false;
				hapticsThread->join();
			}
		}
		void hapticCursor::StartListen()
		{
			runHpatics = true;
			boost::thread t = boost::thread(*deviceWorker);
			hapticsThread = &t;
		}
	}
}
