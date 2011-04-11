#ifndef TF_HISTOGRAM
#define TF_HISTOGRAM

#include <TFCommon.h>

namespace M4D {
namespace GUI {	
namespace TF {

class Histogram{

public:

	typedef boost::shared_ptr<Histogram> Ptr;

	typedef std::vector<Size>::const_iterator const_iterator;

	typedef Size value_type;

	Histogram():
		values_(),
		maxValue_(0.0f),
		avarageValue_(0.0f){
	}

	void add(const Size value){

		values_.push_back(value);

		if(value > maxValue_)
		{
			maxValue_ = value;
		}

		sum_ += value;

		avarageValue_ = sum_/(float)values_.size();
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

	const value_type& operator[](const Size index){

		tfAssert(index < values_.size());
		return values_[index];
	}
	const_iterator begin(){

		return values_.begin();
	}
	const_iterator end(){

		return values_.end();
	}
	float getRelativeValue(const Size index){

		tfAssert(index < values_.size());
		return maxValue_/values_[index];
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