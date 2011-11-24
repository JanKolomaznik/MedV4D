#ifndef TF_HISTOGRAM
#define TF_HISTOGRAM

#include "MedV4D/GUI/TF/TFCommon.h"
#include "MedV4D/GUI/TF/TFMultiDVector.h"

namespace M4D {
namespace GUI {	
namespace TF {

class HistogramInterface{

public:

	typedef boost::shared_ptr<HistogramInterface> Ptr;

	typedef Size value_type;

	virtual ~HistogramInterface(){}

	bool isSealed(){

		return sealed_;
	}

	virtual void seal() = 0;

	virtual void set(const TF::Coordinates& coords, const Size value) = 0;
	virtual const value_type& get(const TF::Coordinates& coords) = 0;

	virtual Size getDomain(const Size dimension) = 0;
	virtual Size getDimension() = 0;

protected:

	bool sealed_;

	HistogramInterface(): sealed_(false){};
};

template<Size dim>
class Histogram: public HistogramInterface{

	typedef MultiDVector<Size, dim> Data;

public:

	typedef boost::shared_ptr< Histogram<dim> > Ptr;

	typedef typename MultiDVector<Size, dim>::const_iterator const_iterator;

	Histogram(std::vector<Size> dimensionSizes):
		values_(dimensionSizes){
	}

	void seal(){

		sealed_ = true;
	}

	void set(const TF::Coordinates& coords, const Size value){

		tfAssert(!sealed_);

		if(sealed_) return;
		values_.value(coords) = value;
	}

	const value_type& get(const TF::Coordinates& coords){

		return values_.value(coords);
	}

	Size getDomain(const Size dimension){

		return values_.size(dimension);
	}

	Size getDimension(){

		return dim;
	}

	const_iterator begin(){

		return values_.begin();
	}

	const_iterator end(){

		return values_.end();
	}

private:

	MultiDVector<Size, dim> values_;
};

}
}
}

#endif	//TF_HISTOGRAM
