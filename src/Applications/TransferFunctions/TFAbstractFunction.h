#ifndef TF_ABSTRACTFUNCTION
#define TF_ABSTRACTFUNCTION

#include <TFTypes.h>

namespace M4D {
namespace GUI {

class TFAbstractFunction{

public:
	virtual ~TFAbstractFunction();

	void operator=(TFAbstractFunction &function);

	TFFunctionType getType() const;

	TFSize getDomain();

	TFColorMapPtr getColorMap();

	void clear();

protected:
	TFFunctionType type_;

	TFColorMapPtr colorMap_;

	TFAbstractFunction();
};

} // namespace GUI
} // namespace M4D

#endif //TF_ABSTRACTFUNCTION