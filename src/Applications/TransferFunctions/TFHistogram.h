#ifndef TF_HISTOGRAM
#define TF_HISTOGRAM

#include "Imaging/Histogram.h"

#include <TFCommon.h>

namespace M4D {
namespace GUI {	
namespace TF {

struct Histogram{	//TODO vector wrapper w/ maximum

	typedef boost::shared_ptr<Histogram> Ptr;

	std::vector<Size> values;
	Size maxValue;
	Size sum;
	float avarageValue;

	Histogram(): values(), maxValue(0), avarageValue(0){}

	void add(Size value){

		values.push_back(value);
		if(value > maxValue) maxValue = value;
		sum += value;
		avarageValue = sum/(float)values.size();
	}

	Size operator[](Size index){

		tfAssert(index < values.size());
		return values[index];
	}

	Size size(){

		return values.size();
	}

	Size maximum(){

		return maxValue;
	}

	float avarage(){

		return avarageValue;
	}
};

}
}
}

#endif	//TF_HISTOGRAM