#ifndef TF_ABSTRACTFUNCTION
#define TF_ABSTRACTFUNCTION

#include <string>
#include "common/Types.h"

typedef std::string TFName;

class QWidget;

static const int FUNCTION_RANGE = 300;
static const int COLOUR_RANGE = 256;

class TFAFunction{

public:
	TFName name;

	TFAFunction(){}
	~TFAFunction(){}

	virtual void adjustByTransferFunction(
		int* pixel,
		int min,
		int max,
		uint32 &width,
		uint32 &height,
		int brightnessRate,
		int contrastRate) = 0;

	virtual void save() = 0;
	virtual void load() = 0;

	QWidget* getPainter(){
		return _painter;
	}

	QWidget* getTools(){
		return _tools;
	}

protected:
	QWidget* _painter;
	QWidget* _tools;
};


#endif //TF_ABSTRACTFUNCTION