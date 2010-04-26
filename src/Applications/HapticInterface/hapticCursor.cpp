#include "hapticCursor.h"

namespace M4D
{
	namespace Viewer
	{
#pragma region hapticDeviceWorker
		hapticCursor::hapticDeviceWorker::hapticDeviceWorker( cGenericHapticDevice* hapticDevice, hapticCursor* supervisor, bool* runHaptic )
		{
			this->hapticDevice = hapticDevice;
			this->supervisor = supervisor;
			this->runHaptic = runHaptic;
			m_clock = new cPrecisionClock();
			m_clock->start();
			count = 0;
		}

		
		void hapticCursor::hapticDeviceWorker::operator()()
		{
			while(*runHaptic)
			{	
				cVector3d hapticPosition;
				hapticDevice->getPosition(hapticPosition);
				supervisor->SetCursorPosition(hapticPosition);
				hapticDevice->setForce(supervisor->GetForce());

				double clock = m_clock->getCurrentTimeSeconds();
				double fps = 1.0 / clock;
				m_clock->reset();
				//std::cout << fps << std::endl;

				//std::cout << supervisor->GetForce().x << " " << supervisor->GetForce().y << " " << supervisor->GetForce().z << " " << std::endl;
			}
		}
#pragma endregion hapticDeviceWorker


		hapticCursor::hapticCursor(vtkImageData *input, vtkRenderWindow* renderWindow, transitionFunction* hapticForceTransitionFunction) : cursorInterface(input)
		{
			handler = new cHapticDeviceHandler();
			deviceWorker = NULL;
			hapticDevice = NULL;
			this->renderWindow = renderWindow;
			this->hapticForceTransitionFunction = hapticForceTransitionFunction;
			runHpatics = false;
		}

		hapticCursor::~hapticCursor()
		{
			stop();
			delete(handler);
			if (deviceWorker != NULL)
				delete(deviceWorker);
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
			boost::mutex::scoped_lock lck(cursorMutex);
			cVector3d lastRealPosition(cursorCenter[0], cursorCenter[1], cursorCenter[2]);

			cVector3d radiusCubeCenterChai(cursorRadiusCubeCenter[0], cursorRadiusCubeCenter[1], cursorRadiusCubeCenter[2]);
			cVector3d realCursorPosition(cursorPosition);

			if (realCursorPosition.x > info.m_workspaceRadius)
			{
				realCursorPosition.x = info.m_workspaceRadius;
			}
			else if (realCursorPosition.x < -info.m_workspaceRadius)
			{
				realCursorPosition.x = -info.m_workspaceRadius;
			}
			if (realCursorPosition.y > info.m_workspaceRadius)
			{
				realCursorPosition.y = info.m_workspaceRadius;
			}
			else if (realCursorPosition.y < -info.m_workspaceRadius)
			{
				realCursorPosition.y = -info.m_workspaceRadius;
			}
			if (realCursorPosition.z > info.m_workspaceRadius)
			{
				realCursorPosition.z = info.m_workspaceRadius;
			}
			else if (realCursorPosition.z < -info.m_workspaceRadius)
			{
				realCursorPosition.z = -info.m_workspaceRadius;
			}

			cVector3d newRealPosition = radiusCubeCenterChai + realCursorPosition / info.m_workspaceRadius * scale / 2.0; // radius gets just half of length of cube edge but scale means full length
			double newRealPositionVTK[3];
			newRealPositionVTK[0] = newRealPosition.x;
			newRealPositionVTK[1] = newRealPosition.y;
			newRealPositionVTK[2] = newRealPosition.z;
			cursorCenter[0] = newRealPositionVTK[0];
			cursorCenter[1] = newRealPositionVTK[1];
			cursorCenter[2] = newRealPositionVTK[2];

			cVector3d newForce = newRealPosition - lastRealPosition;
			if ((newForce.x != 0) || (newForce.y != 0) || (newForce.z != 0))
			{
				newForce.normalize();
			}
			newForce *= -1 * info.m_maxForce;
			int coords[3];
			coords[0] = (newRealPositionVTK[0] - imageRealOffsetWidth) / imageSpacingWidth + imageOffsetWidth;
			coords[1] = (newRealPositionVTK[1] - imageRealOffsetHeight) / imageSpacingHeight + imageOffsetHeight;
			coords[2] = (newRealPositionVTK[2] - imageRealOffsetDepth) / imageSpacingDepth + imageOffsetDepth;

			if ((coords[0] >= imageOffsetWidth) && (coords[0] < (imageOffsetWidth + imageDataWidth)) &&
				(coords[1] >= imageOffsetHeight) && (coords[1] < (imageOffsetHeight + imageDataHeight)) &&
				(coords[2] >= imageOffsetDepth) && (coords[2] < (imageOffsetDepth + imageDataDepth)))
			{
				unsigned short cursorVolumeValue = (unsigned short)input->GetScalarComponentAsDouble(coords[0], coords[1], coords[2], 0);
				newForce *= hapticForceTransitionFunction->GetValueOnPoint(cursorVolumeValue);
				force = newForce;
			}
			else
			{
				force.zero();
			}
		}
		cVector3d& hapticCursor::GetForce()
		{
			boost::mutex::scoped_lock lck(cursorMutex);
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
