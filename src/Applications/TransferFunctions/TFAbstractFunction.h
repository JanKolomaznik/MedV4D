#ifndef TF_ABSTRACTFUNCTION
#define TF_ABSTRACTFUNCTION

#include <string>
#include <map>
#include <vector>

#include <TFTypes.h>


class TFAbstractFunction{

public:
	TFName name;

	TFType getType() const{
		return type_;
	}

	virtual ~TFAbstractFunction(){};

	virtual TFAbstractFunction* clone() = 0;

protected:
	TFType type_;

	TFAbstractFunction(): type_(TFTYPE_UNKNOWN){};
};


template<typename ElementType>
bool adjustByTransferFunction(
	TFAbstractFunction* transferFunction,
	std::vector<ElementType>* result,
	ElementType min,
	ElementType max,
	std::size_t resultSize){

	switch(transferFunction->getType()){
		case TFTYPE_SIMPLE:
		{
			return adjustBySimpleFunction<ElementType>(transferFunction, result, min, max, resultSize);
			break;
		}
		case TFTYPE_UNKNOWN:
		default:
		{
			assert("Unknown Transfer Function");
			break;
		}
	}
	return false;
}

#endif //TF_ABSTRACTFUNCTION