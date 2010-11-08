#ifndef TF_ABSTRACTFUNCTION
#define TF_ABSTRACTFUNCTION

#include <string>
#include <map>
#include <vector>

#include <TFTypes.h>

namespace M4D {
namespace GUI {

class TFAbstractFunction{

public:
	TFType getType() const{
		return type_;
	}

	virtual ~TFAbstractFunction(){};

	virtual TFAbstractFunction* clone() = 0;

protected:
	TFType type_;

	TFAbstractFunction(): type_(TFTYPE_UNKNOWN){};
};

} // namespace GUI
} // namespace M4D

#endif //TF_ABSTRACTFUNCTION