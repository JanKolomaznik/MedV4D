#ifdef OIS_ENABLED

#include "JoyInput.h"
#include <OISException.h>
#include <sstream>
#include <iostream>
#include "MedV4D/Common/Debug.h"

JoyInput::JoyInput() : m_InputManager(NULL), m_bInitialized(false) {

}

JoyInput::~JoyInput() {
}

bool JoyInput::startup(size_t wndHandle) {
	if(m_bInitialized) {
		std::cout << "Input system already initialized" << std::endl;
		return false;
	}
	try {
		OIS::ParamList pl;
		std::ostringstream windowHndStr;
		
		windowHndStr << wndHandle;
		pl.insert(std::make_pair(std::string("WINDOW"), windowHndStr.str()));
		m_InputManager = OIS::InputManager::createInputSystem(pl);
		m_InputManager->enableAddOnFactory(OIS::InputManager::AddOn_All);

		D_COMMAND(
		//Print debugging information
		unsigned int v = m_InputManager->getVersionNumber();
		std::cout << "OIS Version: " << (v>>16 ) << "." << ((v>>8) & 0x000000FF) << "." << (v & 0x000000FF)
			<< "\nRelease Name: " << m_InputManager->getVersionName()
			<< "\nManager: " << m_InputManager->inputSystemName()
			<< "\nTotal Keyboards: " << m_InputManager->getNumberOfDevices(OIS::OISKeyboard)
			<< "\nTotal Mice: " << m_InputManager->getNumberOfDevices(OIS::OISMouse)
			<< "\nTotal JoySticks: " << m_InputManager->getNumberOfDevices(OIS::OISJoyStick) << std::endl;
		)

	} catch(OIS::Exception &e) {
		std::cout << "Error while constructing OIS InputManager: " << e.eText << std::endl;
		return false;
	}
	
	if(m_InputManager->getNumberOfDevices(OIS::OISJoyStick) < 1) {
		std::cout << "No Joysticks found" << std::endl;
		return false;
	}

	try
	{
		int numSticks = m_InputManager->getNumberOfDevices(OIS::OISJoyStick);
		if(numSticks > 0) {
			m_vJoys.resize(numSticks);
			m_vJoyState.resize(numSticks);
		}
		for( int i = 0; i < numSticks; ++i )
		{
			m_vJoys[i] = (OIS::JoyStick*)m_InputManager->createInputObject( OIS::OISJoyStick, false );
			//m_pJoys[i]->setEventCallback( &handler );
			m_vJoys[i]->setBuffered(false);

			D_COMMAND(std::cout << "\nCreating Joystick " << (i + 1) << std::endl);
		}
		D_COMMAND(
		for(int i = 0; i < numSticks; i++) {
			std::cout << "Joystick " << (i) << ": " << m_vJoys[i]->vendor() << std::endl 
			
				<< "\tAxes: " << m_vJoys[i]->getNumberOfComponents(OIS::OIS_Axis) << std::endl 
				<< "\tSliders: " << m_vJoys[i]->getNumberOfComponents(OIS::OIS_Slider) << std::endl 
				<< "\tPOV/HATs: " << m_vJoys[i]->getNumberOfComponents(OIS::OIS_POV) << std::endl 
				<< "\tButtons: " << m_vJoys[i]->getNumberOfComponents(OIS::OIS_Button) << std::endl 
				<< "\tVector3: " << m_vJoys[i]->getNumberOfComponents(OIS::OIS_Vector3) << std::endl;
		}
		)
	}
	catch(OIS::Exception &ex)
	{
		std::cout << "Exception raised on joystick creation: " << ex.eText << std::endl;
	}

	return true;
}

void JoyInput::destroy() {
	//Destroying the manager will cleanup unfreed devices
	std::cout << "Cleaning up...\n";
	if( m_InputManager )
		OIS::InputManager::destroyInputSystem(m_InputManager);
}

void JoyInput::updateJoys() {
	for(int i = 0; i < m_vJoys.size(); i++)
		updateJoy(i);
}

void JoyInput::updateJoy(unsigned int idxJoy) {
	m_vJoys[idxJoy]->capture();

	m_vJoyState[idxJoy] = m_vJoys[idxJoy]->getJoyStickState();
}

int JoyInput::getAxis(unsigned int idxJoy, unsigned int idxAxis) {
	return m_vJoyState[idxJoy].mAxes[idxAxis].abs;
}

int JoyInput::getSlider(unsigned int idxJoy, unsigned int idxSlider) {
	return m_vJoyState[idxJoy].mSliders[idxSlider].abY;
}

bool JoyInput::getButton(unsigned int idxJoy, unsigned int idxButton) {
	return m_vJoyState[idxJoy].mButtons[idxButton];
}

size_t JoyInput::getNrJoysticks() {
	return m_vJoys.size(); 
}

size_t JoyInput::getNrAxes(unsigned int idxJoy) { 
	return m_vJoys[idxJoy]->getNumberOfComponents(OIS::OIS_Axis); 
}

size_t JoyInput::getNrSliders(unsigned int idxJoy) { 
	return m_vJoys[idxJoy]->getNumberOfComponents(OIS::OIS_Slider); 
}

size_t JoyInput::getNrButtons(unsigned int idxJoy) {
	return m_vJoys[idxJoy]->getNumberOfComponents(OIS::OIS_Button); 
}

#endif