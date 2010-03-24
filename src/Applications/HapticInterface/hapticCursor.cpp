#include "hapticCursor.h"

namespace M4D
{
	namespace Viewer
	{
		hapticCursor::hapticDeviceWorker::hapticDeviceWorker(cGenericHapticDevice* hapticDevice, hapticCursor* supervisor, bool* runHaptic)
		{
			this->hapticDevice = hapticDevice;
			this->supervisor = supervisor;
			this->runHaptic = runHaptic;
		}
		void hapticCursor::hapticDeviceWorker::StartListen()
		{
			boost::thread hapticsThread(*this);
			hapticsThread.join();
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

		hapticCursor::hapticCursor(Imaging::InputPortTyped< Imaging::AImage > *inPort) : cursorInterface(inPort)
		{
			handler = new cHapticDeviceHandler();
			runHpatics = true;
		}
		hapticCursor::~hapticCursor()
		{
			stop();
			delete(hapticDevice);
			delete(handler);
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
				deviceWorker->StartListen();
			}
			else
			{
				std::cout << "No haptic device found!" << std::endl;
			}
		}
		cVector3d& hapticCursor::GetForce()
		{
			return force;
		}
		void hapticCursor::stop()
		{
			runHpatics = false;
		}
	}
}
