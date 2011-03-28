#ifndef TF_HISTOGRAM
#define TF_HISTOGRAM

#include <TFCommon.h>

#include <cmath>

namespace M4D {
namespace GUI {	
namespace TF {

class Histogram{	//TODO vector wrapper w/ maximum

public:
	typedef boost::shared_ptr<Histogram> Ptr;

	typedef std::vector<Size>::const_iterator const_iterator;

	typedef Size value_type;

	Histogram():
		values_(),
		maxValue_(0),
		avarageValue_(0),
		logBase_(20),
		logMod_(std::log(logBase_)),
		logMax_(1){
	}

	void add(const Size value){

		values_.push_back(value);

		if(value > maxValue_)
		{
			maxValue_ = value;
			logMax_ = std::log((float)maxValue_)/logMod_;
		}

		sum_ += value;

		avarageValue_ = sum_/(float)values_.size();
	}

	void setLogBase(float logBase){

		logBase_ = logBase;
		logMod_ = std::log(logBase_);
		logMax_ = std::log((float)maxValue_)/logMod_;
	}

	Size size(){

		return values_.size();
	}
	value_type maximum(){

		return maxValue_;
	}
	float avarage(){

		return avarageValue_;
	}
	float logBase(){

		return logBase_;
	}

	const value_type& operator[](Size index){

		tfAssert(index < values_.size());
		return values_[index];
	}
	const_iterator begin(){

		return values_.begin();
	}
	const_iterator end(){

		return values_.end();
	}
	float getRelativeValue(Size index){

		tfAssert(index < values_.size());
		return maxValue_/values_[index];
	}
	float getLogValue(Size index){

		tfAssert(index < values_.size());
		float value = values_[index];
		if(value > 0) value = std::log(value)/logMod_;

		return value;
	}
	float getRelLogValue(Size index){

		tfAssert(index < values_.size());
		return getLogValue(index)/logMax_;
	}

private:

	std::vector<Size> values_;
	Size maxValue_;
	Size sum_;
	float avarageValue_;

	float logBase_;
	float logMod_;
	float logMax_;
};

}
}
}

#endif	//TF_HISTOGRAM