#ifndef M4D_GUI_OPENGLHAPTICVIEWERWIDGET_H_TRANSITION_FUNCTION
#define M4D_GUI_OPENGLHAPTICVIEWERWIDGET_H_TRANSITION_FUNCTION

#include <vector>

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
	transitionFunction(unsigned short minPoint, unsigned short maxPoint, double minValue, double maxValue);
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
};

#endif