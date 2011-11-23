#ifndef TF_MULTI_D_VECTOR
#define TF_MULTI_D_VECTOR

#include "GUI/TF/TFCommon.h"

namespace M4D {
namespace GUI {
namespace TF {

//---template-recursive-declaration---

template<typename Value, Size dim>
class MultiDVector{

	friend class MultiDVector<Value, dim + 1>;

	typedef MultiDVector<Value, dim> MyType;
	typedef std::vector< MultiDVector<Value, dim - 1> > InnerVector;

public:

	typedef boost::shared_ptr<MyType> Ptr;

	typedef MultiDVector<Value, dim - 1> value_type;

	typedef typename InnerVector::iterator iterator;
	typedef typename InnerVector::const_iterator const_iterator;

	MultiDVector(std::vector<Size> dimensionSizes){	

		tfAssert(dimensionSizes.size() == dim);

		dimensionSizes_.swap(dimensionSizes);

		std::vector<Size> nextSizes;
		for(Size i = 1; i < dim; ++i)
		{
			nextSizes.push_back(dimensionSizes_[i]);
		}

		Size mySize = dimensionSizes_[0];
		for(Size i = 0; i < mySize; ++i)
		{
			vector_.push_back(value_type(nextSizes));
		}
	}

	MultiDVector(const MyType& vector){

		const_iterator begin = vector.begin();
		const_iterator end = vector.end();
		for(const_iterator it = begin; it!=end; ++it)
		{
			vector_.push_back(*it);
		}
		
		dimensionSizes_ = vector.dimensionSizes_;
	}

	void operator=(const MyType& vector){

		vector_.clear();

		const_iterator begin = vector.begin();
		const_iterator end = vector.end();
		for(const_iterator it = begin; it!=end; ++it)
		{
			vector_.push_back(*it);
		}
		
		dimensionSizes_ = vector.dimensionSizes_;
	}

	void swap(MyType& vector){

		vector_.swap(vector.vector_);
	}

	void resize(const Size newSize){

		vector_.resize(newSize, value_type(dimensionSizes_));
	}

	Size size(const Size dimension){

		return dimensionSizes_[dimension-1];
	}

	Value& value(const TF::Coordinates& coords){

		tfAssert(coords.size() == dim);
		return vector_[coords[0]].getValue_(coords, dim);
	}

	value_type& operator[](const TF::Size index){

		return vector_[index];
	}

	iterator begin(){

		return vector_.begin();
	}

	iterator end(){

		return vector_.end();
	}

	const_iterator begin() const{

		return vector_.begin();
	}

	const_iterator end() const{

		return vector_.end();
	}

private:

	MultiDVector(){}

	Value& getValue_(const TF::Coordinates& coords, Size realDim){
		
		Size myIndex = coords[realDim - dim];
		return vector_[coords[myIndex]].getValue_(coords, realDim);
	}

	InnerVector vector_;
	std::vector<Size> dimensionSizes_;
};

//---specialization-for-last-dimension---

template<typename Value>
class MultiDVector<Value, 1>{

	friend class MultiDVector<Value, 2>;

	typedef MultiDVector<Value, 1> MyType;
	typedef std::vector<Value> InnerVector;

public:

	typedef boost::shared_ptr<MyType> Ptr;

	typedef Value value_type;

	typedef typename InnerVector::iterator iterator;
	typedef typename InnerVector::const_iterator const_iterator;

	MultiDVector(std::vector<Size> dimensionSizes){	

		tfAssert(dimensionSizes.size() == 1);

		Size mySize = dimensionSizes[0];
		for(Size i = 0; i < mySize; ++i)
		{
			vector_.push_back(value_type());
		}
	}

	MultiDVector(const MyType& vector){

		const_iterator begin = vector.begin();
		const_iterator end = vector.end();
		for(const_iterator it = begin; it!=end; ++it)
		{
			vector_.push_back(*it);
		}
	}

	void operator=(const MyType& vector){

		vector_.clear();

		const_iterator begin = vector.begin();
		const_iterator end = vector.end();
		for(const_iterator it = begin; it!=end; ++it)
		{
			vector_.push_back(*it);
		}
	}

	void swap(MyType& vector){

		vector_.swap(vector.vector_);
	}

	void resize(const Size newSize){

		vector_.resize(newSize);
	}

	Size size(const Size dimension){

		tfAssert(dimension == 1);
		if(dimension != 1) throw std::out_of_range("Bad dimension");
		return vector_.size();
	}

	Value& value(const TF::Coordinates& coords){

		tfAssert(coords.size() == 1);
		return vector_[coords[0]];
	}

	Value& operator[](const TF::Size index){

		return vector_[index];
	}

	iterator begin(){

		return vector_.begin();
	}

	iterator end(){

		return vector_.end();
	}

	const_iterator begin() const{

		return vector_.begin();
	}

	const_iterator end() const{

		return vector_.end();
	}

protected:

	MultiDVector(){}

	Value& getValue_(const TF::Coordinates& coords, Size realDim){
		
		Size myIndex = coords[realDim - 1];
		return vector_[coords[myIndex]];
	}

	InnerVector vector_;
};

}
}
}

#endif	//TF_MULTI_D_VECTOR
