#ifndef TF_HISTOGRAM
#define TF_HISTOGRAM

#include "Imaging/Histogram.h"

#include <TFCommon.h>

namespace M4D {
namespace GUI {	
namespace TF {

class Histogram{	//TODO vector wrapper w/ maximum

public:
	typedef boost::shared_ptr<Histogram> Ptr;

	Histogram(): values_(), maxValue_(0), avarageValue_(0){}

	void add(const Size value){

		values_.push_back(value);
		if(value > maxValue_) maxValue_ = value;
		sum_ += value;
		avarageValue_ = sum_/(float)values_.size();
	}

	Size& operator[](Size index){

		tfAssert(index < values_.size());
		return values_[index];
	}

	Size size(){

		return values_.size();
	}

	Size maximum(){

		return maxValue_;
	}

	float avarage(){

		return avarageValue_;
	}

private:

	std::vector<Size> values_;
	Size maxValue_;
	Size sum_;
	float avarageValue_;
};

}
}
}

#endif	//TF_HISTOGRAM