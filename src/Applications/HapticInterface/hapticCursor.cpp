#include "hapticCursor.h"
#include "boost/bind.hpp"

#define SPRING_POWER 1000.0
#define EPSILON 0.1
#define NUMBER_OF_VECTORS 20
#define MAX_FORCE 14.0

namespace M4D
{
	namespace Viewer
	{

		hapticCursor::hapticCursor(vtkImageData *input, vtkRenderWindow* renderWindow, transitionFunction* hapticForceTransitionFunction) : cursorInterface(input)
		{
			traceLogEnabled = false;
			proxyMode = false;
			handler = new cHapticDeviceHandler();
			hapticDevice = NULL;
			this->renderWindow = renderWindow;
			this->hapticForceTransitionFunction = hapticForceTransitionFunction;
			zoomInButtonPressed = false;
			zoomOutButtonPressed = false;
			runHaptics = false;
			this->hapticDevice = hapticDevice;
			m_clock = new cPrecisionClock();
			count = 0;
			value = 0;
			lastPosition.zero();
			epsilon = EPSILON;
			ksi = 0.3;
			springPower = SPRING_POWER;
			numberOfVectors = NUMBER_OF_VECTORS;
			vectors.clear();
			cVector3d c = cVector3d(0.0, 0.0, 0.0);
			for (int i = 0; i < numberOfVectors; ++i)
			{
				vectors.push_back(c);
			}
		}

		hapticCursor::hapticCursor( vtkRenderWindow* renderWindow, transitionFunction* hapticForceTransitionFunction )
		{
			proxyMode = false;
			traceLogEnabled = false;
			handler = new cHapticDeviceHandler();
			hapticDevice = NULL;
			this->renderWindow = renderWindow;
			this->hapticForceTransitionFunction = hapticForceTransitionFunction;
			zoomInButtonPressed = false;
			zoomOutButtonPressed = false;
			runHaptics = false;
			this->hapticDevice = hapticDevice;
			m_clock = new cPrecisionClock();
			count = 0;
			value = 0;
			lastPosition.zero();
			epsilon = EPSILON;
			ksi = 0.3;
			springPower = SPRING_POWER;
			numberOfVectors = NUMBER_OF_VECTORS;
			vectors.clear();
			cVector3d c = cVector3d(0.0, 0.0, 0.0);
			for (int i = 0; i < numberOfVectors; ++i)
			{
				vectors.push_back(c);
			}
		}
		hapticCursor::~hapticCursor()
		{
			if (traceLogEnabled)
			{
				traceLogFile.close();
			}
			stop();
			m_clock->stop();
			delete(m_clock);
			delete(hapticsThread);
			delete(handler);
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

			if(!proxyMode)
			{
				cVector3d radiusCubeCenterChai(cursorRadiusCubeCenter[0], cursorRadiusCubeCenter[1], cursorRadiusCubeCenter[2]);
				cVector3d realCursorPosition(cursorPosition.y, cursorPosition.z, cursorPosition.x);

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
				if (traceLogEnabled && ((newRealPositionVTK[0] != cursorCenter[0]) || (newRealPositionVTK[1] != cursorCenter[1]) || (newRealPositionVTK[2] != cursorCenter[2])))
				{
					traceLogFile << newRealPositionVTK[0] << " " << newRealPositionVTK[1] << " " << newRealPositionVTK[2] << std::endl; 
				}
				cursorCenter[0] = newRealPositionVTK[0];
				cursorCenter[1] = newRealPositionVTK[1];
				cursorCenter[2] = newRealPositionVTK[2];

				cVector3d newForce = cursorPosition - lastPosition;
				for (int i = 0; i < numberOfVectors - 1; ++i)
				{
					vectors[i] = vectors[i + 1];
				}
				vectors[numberOfVectors - 1] = newForce;
				lastPosition = cursorPosition;
				if ((newForce.x != 0) || (newForce.y != 0) || (newForce.z != 0))
				{
					newForce.normalize();
				}
				newForce *= -1 * MAX_FORCE;
				int coords[3];
				coords[0] = (newRealPositionVTK[0] - imageRealOffsetWidth) / imageSpacingWidth + imageOffsetWidth;
				coords[1] = (newRealPositionVTK[1] - imageRealOffsetHeight) / imageSpacingHeight + imageOffsetHeight;
				coords[2] = (newRealPositionVTK[2] - imageRealOffsetDepth) / imageSpacingDepth + imageOffsetDepth;

				if ((coords[0] >= imageOffsetWidth) && (coords[0] < (imageOffsetWidth + imageDataWidth)) &&
					(coords[1] >= imageOffsetHeight) && (coords[1] < (imageOffsetHeight + imageDataHeight)) &&
					(coords[2] >= imageOffsetDepth) && (coords[2] < (imageOffsetDepth + imageDataDepth)))
				{
					unsigned short cursorVolumeValue = (unsigned short)input->GetScalarComponentAsDouble(coords[0], coords[1], coords[2], 0);
					value = cursorVolumeValue;
					if (hapticForceTransitionFunction->GetSolidFrom() != -1)
					{
						if (hapticForceTransitionFunction->GetSolidFrom() < cursorVolumeValue)
						{
							solidPlaneParams.zero();
							for (int i = 0; i < numberOfVectors; ++i)
							{
								solidPlaneParams += vectors[i];
							}
							if ((solidPlaneParams.x != 0) || (solidPlaneParams.y != 0) || (solidPlaneParams.z != 0))
							{
								solidPlaneParams.normalize();
								proxyMode = true;
								dParamOfPlane = -(solidPlaneParams * cursorPosition);
							}
						}
					}
					if (hapticForceTransitionFunction->GetSolidTo() != -1)
					{
						if (hapticForceTransitionFunction->GetSolidTo() > cursorVolumeValue)
						{
							solidPlaneParams.zero();
							for (int i = 0; i < numberOfVectors; ++i)
							{
								solidPlaneParams += vectors[i];
							}
							if ((solidPlaneParams.x != 0) || (solidPlaneParams.y != 0) || (solidPlaneParams.z != 0))
							{
								solidPlaneParams.normalize();
								proxyMode = true;
								dParamOfPlane = -(solidPlaneParams * cursorPosition);
							}
						}
					}
					double valueOnPoint = hapticForceTransitionFunction->GetValueOnPoint(cursorVolumeValue);
					cVector3d hlp;
					hlp.zero();
					for (int i = 0; i < numberOfVectors; ++i)
					{
						hlp += vectors[i];
					}
					if ((hlp.x != 0) || (hlp.y != 0) || (hlp.z != 0))
					{
						hlp.normalize();
					}
					hlp *= valueOnPoint * -1.0 * MAX_FORCE;
					cVector3d delta = hlp - force;
					if (delta.length() > epsilon)
					{
					delta.normalize();
					delta *= epsilon;
					}
					hlp = force + delta;
					/*if (delta.x > epsilon)
					delta.x = epsilon;
					if (delta.x < -epsilon)
					delta.x = -epsilon;
					if (delta.y > epsilon)
					delta.y = epsilon;
					if (delta.y < -epsilon)
					delta.y = -epsilon;
					if (delta.z > epsilon)
					delta.z = epsilon;
					if (delta.z < -epsilon)
					delta.z = -epsilon;*/
					//double maxForce = ( valueOnPoint + ksi ) * info.m_maxForce;
					//double minForce = ( valueOnPoint - ksi ) > 0 ? ( valueOnPoint - ksi ) * info.m_maxForce : 0;
					//if (hlp.length() > maxForce)
					//{
					//	hlp.normalize();
					//	hlp *= maxForce;
					//}
					//if (hlp.length() < minForce)
					//{
					//	hlp.normalize();
					//	hlp *= minForce;
					//}
					force = hlp;
				}
				else
				{
					force.zero();
				}
			}
			else
			{
				if (((cursorPosition * solidPlaneParams + dParamOfPlane) >= 0) && ((hapticForceTransitionFunction->GetSolidFrom() != -1) || (hapticForceTransitionFunction->GetSolidTo() != -1)))
				{
					cVector3d forceDirection = lastPosition - cursorPosition;
					double forcePower = forceDirection.length() * springPower;
					if (forcePower > MAX_FORCE)
					{
						forcePower = MAX_FORCE;
					}
					forceDirection.normalize();
					force = forceDirection * forcePower;
				}
				else
				{
					proxyMode = false;
				}
			}
		}
		cVector3d& hapticCursor::GetForce()
		{
			boost::mutex::scoped_lock lck(cursorMutex);
			return force;
		}
		void hapticCursor::stop()
		{
			runHaptics = false;
			hapticsThread->join();
		}
		void hapticCursor::StartListen()
		{
			m_clock->start();
			runHaptics = true;
			hapticsThread = new boost::thread(boost::bind(&hapticCursor::deviecWorker, this));
		}

		void hapticCursor::SetZoomInButtonPressed( bool pressed )
		{
			boost::mutex::scoped_lock lck(cursorMutex);
			if (!pressed && zoomInButtonPressed)
			{
				SetScale((scale / 3.0) * 2.0);
			}
		}

		void hapticCursor::SetZoomOutButtonPressed( bool pressed )
		{
			boost::mutex::scoped_lock lck(cursorMutex);
			if (!pressed && zoomOutButtonPressed)
			{
				SetScale((scale / 2.0) * 3.0);
			}
		}

		void hapticCursor::deviecWorker()
		{
			while (runHaptics)
			{	
				cSleepMs(1); // may be performance issue
				cVector3d hapticPosition;
				hapticDevice->getPosition(hapticPosition);
				//std::cout << hapticPosition << std::endl;
				SetCursorPosition(hapticPosition);
				hapticDevice->setForce(GetForce());
				bool zoomIn = false;
				bool zoomOut = false;

				hapticDevice->getUserSwitch(0, zoomIn);
				hapticDevice->getUserSwitch(1, zoomOut);

				SetZoomInButtonPressed(zoomIn);
				SetZoomOutButtonPressed(zoomOut);

				//double clock = m_clock->getCurrentTimeSeconds();
				//double fps = 1.0 / clock;
				//m_clock->reset();
				//std::cout << fps << std::endl;

				std::cout << GetForce() << std::endl;
			}
		}

		int hapticCursor::GetValue()
		{
			boost::mutex::scoped_lock lck(cursorMutex);
			return value;
		}

		void hapticCursor::SetTraceLogOn( string file )
		{
			boost::mutex::scoped_lock lck(cursorMutex);
			traceLogFile.open(file.c_str());
			traceLogEnabled = true;
		}

		void hapticCursor::SetTraceLogOff()
		{
			boost::mutex::scoped_lock lck(cursorMutex);
			traceLogFile.close();
			traceLogEnabled = false;	
		}
	}
}
