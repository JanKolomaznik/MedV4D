#ifndef _PROGRESS_CALLBACK_
#define _PROGRESS_CALLBACK_

namespace viewer {

class CProgress {
public:
	virtual void UpdateProgress(float fraction) = 0;
};

class CTextProgress: public CProgress {
public:
	void UpdateProgress(float fraction) {
		printf("Progress: %.2f%\n", fraction*100);
	}
};

}

#endif