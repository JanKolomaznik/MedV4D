#ifndef M4D_GUI_OPENGLHAPTICVIEWERWIDGET_H_TRANSITION_FUNCTION
#define M4D_GUI_OPENGLHAPTICVIEWERWIDGET_H_TRANSITION_FUNCTION

#include <vector>
#include "Imaging/Imaging.h"

class transitionFunction // class for function which has some values on some points and computes lineary between these points
{
public:
	double GetValueOnPoint(unsigned short point);
	void SetValueOnPoint(unsigned short point, double val);
	double GetValueOfMaxPoint();
	double GetValueOfMinPoint();
	void SetValueOfMaxPoint(double val);
	void SetValueOfMinPoint(double val);
	unsigned short GetMinPoint();
	void SetMinPoint(unsigned short point);
	unsigned short GetMaxPoint();
	void SetMaxPoint(unsigned short point);
	void SetPointOnPosition(std::size_t pos, unsigned short point);
	void DeletePointOnPosition(std::size_t position);
	std::size_t size();
	unsigned short GetPointOnPosition(std::size_t pos);
	void Reset(unsigned short minPoint, unsigned short maxPoint, double minValue, double maxValue);
	transitionFunction(unsigned short minPoint, unsigned short maxPoint, double minValue, double maxValue);
	transitionFunction(){}
protected:

	struct pointValue
	{
	public:
		pointValue(unsigned short point, double val)
		{
			this->point = point;
			this->val = val;
		}
		bool operator==(const pointValue& rhs);
		bool operator!=(const pointValue& rhs);
		bool operator>(const pointValue& rhs);
		bool operator<(const pointValue& rhs);
		bool operator>=(const pointValue& rhs);
		bool operator<=(const pointValue& rhs);
		double val;
		unsigned short point;
	};
	std::vector< pointValue > data;
	boost::mutex accesMutex;
};

#endif