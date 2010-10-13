#ifndef TF_APPLICATOR
#define TF_APPLICATOR

#include <string>
#include <map>
#include <vector>

#include <TFTypes.h>

#include <TFSimpleFunction.h>


struct TFApplicator{

	template<typename ElementIterator>
	static bool apply(
		TFAbstractFunction* transferFunction,
		ElementIterator result,
		TFSize inputRange,
		TFSize outputRange){

		switch(transferFunction->getType()){
			case TFTYPE_SIMPLE:
			{
				TFSimpleFunction* simpleFunction = dynamic_cast<TFSimpleFunction*>(transferFunction);
				return simpleFunction->apply<ElementIterator>(result, inputRange, outputRange);
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
};

#endif //TF_APPLICATOR