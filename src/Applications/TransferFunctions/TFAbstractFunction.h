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
		return _type;
	}

	virtual ~TFAbstractFunction(){};

	virtual TFAbstractFunction* clone() = 0;

protected:
	TFType _type;

	TFAbstractFunction(): _type(TFTYPE_UNKNOWN){};
};

#endif //TF_ABSTRACTFUNCTION