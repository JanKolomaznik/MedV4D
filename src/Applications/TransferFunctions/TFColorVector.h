#ifndef TF_MULTIDIMVECTOR
#define TF_MULTIDIMVECTOR

#include <TFCommon.h>
#include <TFColor.h>

namespace M4D {
namespace GUI {
namespace TF {

//---template-recursive-declaration---

template<Size dim>
class ColorVector{

	friend class ColorVector<dim + 1>;

	typedef typename ColorVector<dim> MyType;
	typedef typename std::vector<ColorVector<dim - 1>> InnerVector;

public:

	typedef typename boost::shared_ptr<MyType> Ptr;

	typedef typename ColorVector<dim - 1> value_type;

	typedef typename InnerVector::iterator iterator;
	typedef typename InnerVector::const_iterator const_iterator;

	ColorVector(std::vector<Size> dimensionSizes){	

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

	ColorVector(const MyType& vector){

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

	MyType operator+(const MyType& right){

		MyType result(*this);
		result += right;
		return result;
	}

	void operator+=(const MyType& vector){

		for(Size i = 0; i < vector_.size(); ++i)
		{
			vector_[i] += vector.vector_[i];
		}
	}

	MyType operator-(const MyType& right){

		MyType result(*this);
		result -= right;
		return result;
	}

	void operator-=(const MyType& vector){

		for(Size i = 0; i < vector_.size(); ++i)
		{
			vector_[i] -= vector.vector_[i];
		}
	}

	MyType operator*(const float& value){

		MyType result(*this);
		result *= value;
		return result;
	}

	void operator*=(const float& value){

		for(Size i = 0; i < vector_.size(); ++i)
		{
			vector_[i] *= value;
		}
	}

	MyType operator/(const float& value){

		MyType result(*this);
		result /= value;
		return result;
	}

	void operator/=(const float& value){

		for(Size i = 0; i < vector_.size(); ++i)
		{
			vector_[i] /= value;
		}
	}

	void resize(const Size newSize){

		vector_.resize(newSize, value_type(dimensionSizes_));
	}

	void recalculate(std::vector<Size> dimensionSizes){

		tfAssert(dimensionSizes.size() == dim);

		dimensionSizes_.swap(dimensionSizes);

		std::vector<Size> nextSizes;
		for(Size i = 1; i < dim; ++i)
		{
			nextSizes.push_back(dimensionSizes_[i]);
		}

		int inputSize = vector_.size();
		int outputSize = dimensionSizes_[0];

		float oiRatio = outputSize/(float)inputSize;

		if(oiRatio == 1)	//do not recalculate this dimension
		{
			for(TF::Size i = 1; i <= dim; ++i)
			{
				vector_[i].recalculate(nextSizes);
			}
			return;
		}

		MyType resized;
		if(oiRatio > 1)	//copy 1 index into more
		{
			for(TF::Size i = 1; i <= dim; ++i)	//recalculate all inner dimensions before this dimension
			{
				vector_[i].recalculate(nextSizes);
			}

			int ratio = (int)(oiRatio);	//how many new values respond to 1 old value
			float correction = oiRatio - ratio;
			float corrStep = correction;

			int outputIndexer = 0;
			for(int inputIndexer = 0; inputIndexer < inputSize; ++inputIndexer)
			{
				TF::Size valueCount = ratio + (int)correction;
				for(TF::Size i = 0; i < valueCount; ++i)
				{
					//tfAssert(outputIndexer < outputSize);
					if(outputIndexer >= outputSize) break;

					resized.vector_.push_back(vector_[inputIndexer]);

					++outputIndexer;
				}
				correction -= (int)correction;
				correction += corrStep;
			}
		}
		else	//compute avarage from more indexes into 1
		{
			resized.resize(outputSize);
			float correction = inputSize/(float)outputSize;
			int ratio =  (int)(correction);	//how many old values are used for computing 1 resized values
			correction -= ratio;
			float corrStep = correction;

			int inputIndexer = 0;
			for(int outputIndexer = 0; outputIndexer < outputSize; ++outputIndexer)
			{
				TF::Color computedValue;
				TF::Size valueCount = ratio + (int)correction;
				for(TF::Size i = 0; i < valueCount; ++i)
				{
					//tfAssert(inputIndexer < inputSize);
					if(inputIndexer >= inputSize)
					{
						valueCount = i;
						break;
					}

					resized.vector_[outputIndexer] += vector_[inputIndexer];

					++inputIndexer;
				}
				correction -= (int)correction;
				correction += corrStep;

				if(valueCount == 0) continue;
				resized.vector_[outputIndexer] /= valueCount;
			}

			for(TF::Size i = 1; i <= dim; ++i)	//recalculate all inner dimensions after this dimension
			{
				resized.vector_[i].recalculate(nextSizes);
			}
		}
		vector_.swap(resized.vector_);
	}

	Size size(const Size dimension){

		return dimensionSizes_[dimension-1];
	}

	Color& value(const TF::Coordinates& coords){

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

	ColorVector(){}

	Color& getValue_(const TF::Coordinates& coords, Size realDim){
		
		Size myIndex = coords[realDim - dim];
		return vector_[coords[myIndex]].getValue_(coords, realDim);
	}

	InnerVector vector_;
	std::vector<Size> dimensionSizes_;
};

//---specialization-for-last-dimension---

template<>
class ColorVector<1>{

	friend class ColorVector<2>;

	typedef ColorVector<1> MyType;
	typedef std::vector<Color> InnerVector;

public:

	typedef boost::shared_ptr<MyType> Ptr;

	typedef Color value_type;

	typedef InnerVector::iterator iterator;
	typedef InnerVector::const_iterator const_iterator;

	ColorVector(std::vector<Size> dimensionSizes){	

		tfAssert(dimensionSizes.size() == 1);

		Size mySize = dimensionSizes[0];
		for(Size i = 0; i < mySize; ++i)
		{
			vector_.push_back(value_type());
		}
	}

	ColorVector(const MyType& vector){

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

	MyType operator+(const MyType& right){

		MyType result(*this);
		result += right;
		return result;
	}

	void operator+=(const MyType& vector){

		for(Size i = 0; i < vector_.size(); ++i)
		{
			vector_[i] += vector.vector_[i];
		}
	}

	MyType operator-(const MyType& right){

		MyType result(*this);
		result -= right;
		return result;
	}

	void operator-=(const MyType& vector){

		for(Size i = 0; i < vector_.size(); ++i)
		{
			vector_[i] -= vector.vector_[i];
		}
	}

	MyType operator*(const float& value){

		MyType result(*this);
		result *= value;
		return result;
	}

	void operator*=(const float& value){

		for(Size i = 0; i < vector_.size(); ++i)
		{
			vector_[i] *= value;
		}
	}

	MyType operator/(const float& value){

		MyType result(*this);
		result /= value;
		return result;
	}

	void operator/=(const float& value){

		for(Size i = 0; i < vector_.size(); ++i)
		{
			vector_[i] /= value;
		}
	}

	void resize(const Size newSize){

		vector_.resize(newSize);
	}

	void recalculate(std::vector<Size> dimensionSizes){

		tfAssert(dimensionSizes.size() == 1);

		int inputSize = vector_.size();
		int outputSize = dimensionSizes[0];

		float oiRatio = outputSize/(float)inputSize;

		if(oiRatio == 1) return;	//do not recalculate this dimension

		MyType resized;
		if(oiRatio > 1)	//copy 1 index into more
		{
			int ratio = (int)(oiRatio);	//how many new values respond to 1 old value
			float correction = oiRatio - ratio;
			float corrStep = correction;

			int outputIndexer = 0;
			for(int inputIndexer = 0; inputIndexer < inputSize; ++inputIndexer)
			{
				TF::Size valueCount = ratio + (int)correction;
				for(TF::Size i = 0; i < valueCount; ++i)
				{
					//tfAssert(outputIndexer < outputSize);
					if(outputIndexer >= outputSize) break;

					resized.vector_.push_back(vector_[inputIndexer]);

					++outputIndexer;
				}
				correction -= (int)correction;
				correction += corrStep;
			}
		}
		else	//compute avarage from more indexes into 1
		{
			resized.resize(outputSize);
			float correction = inputSize/(float)outputSize;
			int ratio =  (int)(correction);	//how many old values are used for computing 1 resized values
			correction -= ratio;
			float corrStep = correction;

			int inputIndexer = 0;
			for(int outputIndexer = 0; outputIndexer < outputSize; ++outputIndexer)
			{
				TF::Color computedValue;
				TF::Size valueCount = ratio + (int)correction;
				for(TF::Size i = 0; i < valueCount; ++i)
				{
					//tfAssert(inputIndexer < inputSize);
					if(inputIndexer >= inputSize)
					{
						valueCount = i;
						break;
					}

					resized.vector_[outputIndexer] += vector_[inputIndexer];

					++inputIndexer;
				}
				correction -= (int)correction;
				correction += corrStep;

				if(valueCount == 0) continue;
				resized.vector_[outputIndexer] /= valueCount;
			}
		}
		vector_.swap(resized.vector_);
	}

	Size size(const Size dimension){

		tfAssert(dimension == 1);
		if(dimension != 1) throw std::out_of_range("Bad dimension");
		return vector_.size();
	}

	Color& value(const TF::Coordinates& coords){

		tfAssert(coords.size() == 1);
		return vector_[coords[0]];
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

	ColorVector(){}

	Color& getValue_(const TF::Coordinates& coords, Size realDim){
		
		Size myIndex = coords[realDim - 1];
		return vector_[coords[myIndex]];
	}

	InnerVector vector_;
};

}
}
}

#endif	//TF_MULTIDIMVECTOR