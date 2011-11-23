#ifndef JOY_INPUT_H
#define JOY_INPUT_H

#ifdef OIS_ENABLED
#include <OISInputManager.h>
#include <OISJoyStick.h>
#include <vector>


class JoyInput {
public:
	bool startup(size_t wndHandle) ;
	void destroy() ;
	void updateJoys() ;
	void updateJoy(unsigned int idxJoy) ;

	size_t getNrJoysticks() ;
	size_t getNrAxes(unsigned int idxJoy) ;
	size_t getNrSliders(unsigned int idxJoy) ;
	size_t getNrButtons(unsigned int idxJoy) ;

public:

	OIS::InputManager *m_InputManager;
	std::vector<OIS::JoyStick*> m_vJoys;
	std::vector<OIS::JoyStickState> m_vJoyState;
	bool			m_bInitialized;

	int getAxis(unsigned int idxJoy, unsigned int idxAxis);
	int getSlider(unsigned int idxJoy, unsigned int idxSlider);
	bool getButton(unsigned int idxJoy, unsigned int idxButton);

	JoyInput();
	~JoyInput();
};

#endif

#endif