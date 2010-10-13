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

	TFSimpleFunction(unsigned functionRange = 500, unsigned colorRange = 280);
	TFSimpleFunction(TFName functionName, unsigned functionRange = 500, unsigned colorRange = 280);
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

	void recalculate(unsigned functionRange, unsigned colorRange);

	template<typename ElementIterator>
	bool apply(
		ElementIterator result,
		TFSize inputRange,
		TFSize outputRange){

		if(points_.empty())
		{
			return false;
		}

		double colorRange_double = (double)colorRange_;

		double interval = inputRange/(double)functionRange_;
		TFSize intervalBottom = 0;
		TFSize intervalTop = 0;
		double intervalCorrection = 0;

		unsigned int lastFunctionIndexUsed = 0;
		unsigned int lastResultIndexUsed = 0;

		for (unsigned int i = 1; i < functionRange_; ++i )
		{

			double newTop = intervalTop + interval + intervalCorrection;
			intervalTop = (TFSize)(newTop);
			intervalCorrection = newTop - (double)intervalTop;

			if(intervalTop > intervalBottom)
			{
				std::iterator_traits<ElementIterator>::value_type bottomValue = (std::iterator_traits<ElementIterator>::value_type)((points_[lastFunctionIndexUsed]/colorRange_double)*outputRange);
				std::iterator_traits<ElementIterator>::value_type topValue = (std::iterator_traits<ElementIterator>::value_type)((points_[i]/colorRange_double)*outputRange);

				lastFunctionIndexUsed = i;

				TFSize intervalRange = intervalTop - intervalBottom;
				double step = (topValue - bottomValue)/(double)intervalRange;

				for(unsigned int j = 0; j < intervalRange; ++j)
				{
					result[lastResultIndexUsed + j] = bottomValue + j*step;
				}
				lastResultIndexUsed = lastResultIndexUsed + intervalRange;

				intervalBottom = intervalTop;
			}
		}

		std::iterator_traits<ElementIterator>::value_type correctionValue = (std::iterator_traits<ElementIterator>::value_type)((points_[lastFunctionIndexUsed]/colorRange_double)*outputRange);
		TFSize remainingRange = inputRange - intervalBottom;
		for(unsigned j = 0; j < remainingRange; ++j)
		{
			result[lastResultIndexUsed + j] = correctionValue;
		}

		return true;
	}
	
private:	
	TFPointMap points_;

	unsigned functionRange_;
	unsigned colorRange_;
};

#endif //TF_SIMPLEFUNCTION