#ifndef TF_SIMPLEFUNCTION
#define TF_SIMPLEFUNCTION

#include <map>
#include <vector>
#include <algorithm>
#include <string>
#include <fstream>
#include <cassert>

#include <TFAbstractFunction.h>


#ifndef ROUND
#define ROUND(a) ( (int)(a+0.5) )
#endif


class TFSimpleFunction: public TFAbstractFunction{

public:

	TFSimpleFunction(int functionRange = 500, int colorRange = 280);
	TFSimpleFunction(TFName functionName, int functionRange = 500, int colorRange = 280);
	TFSimpleFunction(TFSimpleFunction &function);

	~TFSimpleFunction();

	void operator=(TFSimpleFunction &function);

	TFAbstractFunction* clone();

	void addPoint(int x, int y);
	void addPoint(TFPoint point);

	void addPoints(TFPoints points);

	void setPoints(TFPointMap points);

	void clear();

	TFPoint getPoint(int coordX);

	TFPoints getAllPoints();

	TFPointMap getPointMap();

	unsigned getFunctionRange();

	unsigned getColorRange();

	void recalculate(int functionRange, int colorRange);
	
private:	
	TFPointMap points_;

	int functionRange_;
	int colorRange_;
};


/*
template<typename ElementType>
bool adjustBySimpleFunction(
	TFAbstractFunction* transferFunction,
	std::vector<ElementType>* result,
	ElementType min,
	ElementType max,
	std::size_t resultSize){

	return false;
}
*/

template<typename ElementType>
inline bool adjustBySimpleFunction(
	TFAbstractFunction* transferFunction,
	std::vector<ElementType>* result,
	ElementType min,
	ElementType max,
	std::size_t resultSize){

	TFSimpleFunction *tf = dynamic_cast<TFSimpleFunction*>(transferFunction);

	if ( !tf)
	{
		return false;
	}

	TFPointMap points = tf->getPointMap();

	if(points.empty())
	{
		return false;
	}

	unsigned functionRange = tf->getFunctionRange();
	double colorRange_double = (double)tf->getColorRange();
	int resultRange = (int)max - min;

	double interval = resultSize/(double)functionRange;
	std::size_t intervalBottom = 0;
	std::size_t intervalTop = 0;
	double intervalCorrection = 0;

	unsigned lastFunctionIndexUsed = 0;
	unsigned lastResultIndexUsed = 0;

	for (unsigned i = 1; i < functionRange; ++i )
	{

		double newTop = intervalTop + interval + intervalCorrection;
		intervalTop = (std::size_t)(newTop);
		intervalCorrection = newTop - (double)intervalTop;

		if(intervalTop > intervalBottom)
		{
			ElementType bottomValue = (ElementType)(ROUND((points[lastFunctionIndexUsed]/colorRange_double)*resultRange)) + min;
			ElementType topValue = (ElementType)(ROUND((points[i]/colorRange_double)*resultRange)) + min;

			lastFunctionIndexUsed = i;

			std::size_t intervalRange = intervalTop - intervalBottom;
			double step = (topValue - bottomValue)/(double)intervalRange;

			for(unsigned j = 0; j < intervalRange; ++j)
			{
				(*result)[lastResultIndexUsed + j] = bottomValue + j*step;
			}
			lastResultIndexUsed = lastResultIndexUsed + intervalRange;

			intervalBottom = intervalTop;
		}
	}

	ElementType correctionValue = (ElementType)(ROUND((points[lastFunctionIndexUsed]/colorRange_double)*resultRange)) + min;
	std::size_t remainingRange = resultSize - intervalBottom;
	for(unsigned j = 0; j < remainingRange; ++j)
	{
		(*result)[lastResultIndexUsed + j] = correctionValue;
	}

	return true;
}

#endif //TF_SIMPLEFUNCTION