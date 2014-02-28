#ifndef TF_COLORVECTOR
#define TF_COLORVECTOR

#include "MedV4D/GUI/TF/Common.h"
#include "MedV4D/GUI/TF/MultiDVector.h"
#include "MedV4D/GUI/TF/Color.h"

namespace M4D {
namespace GUI {
namespace TF {

//---template-recursive-declaration---

template<Size dim>
class ColorVector: public MultiDVector<Color, dim>{
	
	typedef ColorVector<dim> MyType;

public:

	typedef boost::shared_ptr<MyType> Ptr;

	ColorVector(std::vector<Size> dimensionSizes):
		MultiDVector<Color, dim>(dimensionSizes){
	}

	ColorVector(const MyType& vector):
		MultiDVector<Color, dim>(vector) {
	}

	MyType operator+(const MyType& right){

		MyType result(*this);
		result += right;
		return result;
	}

	void operator+=(const MyType& vector){

		for(Size i = 0; i < this->vector_.size(); ++i)
		{
			this->vector_[i] += vector.vector_[i];
		}
	}

	MyType operator-(const MyType& right){

		MyType result(*this);
		result -= right;
		return result;
	}

	void operator-=(const MyType& vector){

		for(Size i = 0; i < this->vector_.size(); ++i)
		{
			this->vector_[i] -= vector.vector_[i];
		}
	}

	MyType operator*(const float& value){

		MyType result(*this);
		result *= value;
		return result;
	}

	void operator*=(const float& value){

		for(Size i = 0; i < this->vector_.size(); ++i)
		{
			this->vector_[i] *= value;
		}
	}

	MyType operator/(const float& value){

		MyType result(*this);
		result /= value;
		return result;
	}

	void operator/=(const float& value){

		for(Size i = 0; i < this->vector_.size(); ++i)
		{
			this->vector_[i] /= value;
		}
	}

	void recalculate(std::vector<Size> dimensionSizes){

		tfAssert(dimensionSizes.size() == dim);

		this->dimensionSizes_.swap(dimensionSizes);

		std::vector<Size> nextSizes;
		for(Size i = 1; i < dim; ++i)
		{
			nextSizes.push_back(this->dimensionSizes_[i]);
		}

		int inputSize = this->vector_.size();
		int outputSize = this->dimensionSizes_[0];

		float oiRatio = outputSize/(float)inputSize;

		if(oiRatio == 1)	//do not recalculate this dimension
		{
			for(TF::Size i = 1; i <= dim; ++i)
			{
				this->vector_[i].recalculate(nextSizes);
			}
			return;
		}

		MyType resized;
		if(oiRatio > 1)	//copy 1 index into more
		{
			for(TF::Size i = 1; i <= dim; ++i)	//recalculate all inner dimensions before this dimension
			{
				this->vector_[i].recalculate(nextSizes);
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

					resized.vector_.push_back(this->vector_[inputIndexer]);

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
				TF::Size valueCount = ratio + (int)correction;
				for(TF::Size i = 0; i < valueCount; ++i)
				{
					//tfAssert(inputIndexer < inputSize);
					if(inputIndexer >= inputSize)
					{
						valueCount = i;
						break;
					}

					resized.vector_[outputIndexer] += this->vector_[inputIndexer];

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
		this->vector_.swap(resized.vector_);
	}

private:

	ColorVector():
	   MultiDVector<Color, dim>(){
	}
};

//---specialization-for-last-dimension---

template<>
class ColorVector<1>: public MultiDVector<Color, 1>{
	
	typedef ColorVector<1> MyType;

public:

	typedef boost::shared_ptr<MyType> Ptr;

	ColorVector(std::vector<Size> dimensionSizes):
		MultiDVector<Color, 1>(dimensionSizes){
	}

	ColorVector(const MyType& vector):
		MultiDVector<Color, 1>(vector){
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

		for(Size i = 0; i < this->vector_.size(); ++i)
		{
			this->vector_[i] /= value;
		}
	}

	void recalculate(std::vector<Size> dimensionSizes){

		tfAssert(dimensionSizes.size() == 1);

		int inputSize = static_cast<int>(this->vector_.size());
		int outputSize = static_cast<int>(dimensionSizes[0]);

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

					resized.vector_.push_back(this->vector_[inputIndexer]);

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
				TF::Size valueCount = ratio + (int)correction;
				for(TF::Size i = 0; i < valueCount; ++i)
				{
					//tfAssert(inputIndexer < inputSize);
					if(inputIndexer >= inputSize)
					{
						valueCount = i;
						break;
					}

					resized.vector_[outputIndexer] += this->vector_[inputIndexer];

					++inputIndexer;
				}
				correction -= (int)correction;
				correction += corrStep;

				if(valueCount == 0) continue;
				resized.vector_[outputIndexer] /= valueCount;
			}
		}
		this->vector_.swap(resized.vector_);
	}

private:

	ColorVector():
	   MultiDVector<Color, 1>(){
	}
};

}
}
}

#endif	//TF_COLORVECTOR
