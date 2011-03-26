#ifndef TF_HOLDERINTERFACE
#define TF_HOLDERINTERFACE

#include <QtGui/QDockWidget>

#include <TFCommon.h>
#include <TFHistogram.h>
#include <TFPaletteButton.h>

#include <TFAbstractFunction.h>
#include <TFAbstractModifier.h>
#include <TFAbstractPainter.h>
#include <TFWorkCopy.h>

#include <TFAdaptation.h>

namespace M4D {
namespace GUI {	

class TFHolderInterface{

public:

	typedef boost::shared_ptr<TFHolderInterface> Ptr;

	struct Info{
		std::string holder;
		std::string predefined;
		std::string function;
		std::string painter;
		std::string modifier;

		Info():
			holder("Unknown"),
			predefined("Unknown"),
			function("Unknown"),
			painter("Unknown"),
			modifier("Unknown"){
		}

		Info(std::string holder, std::string predefined, std::string function, std::string painter, std::string modifier):
			holder(holder),
			predefined(predefined),
			function(function),
			painter(painter),
			modifier(modifier){
		}
	};

	virtual ~TFHolderInterface(){}

	virtual void save() = 0;
	virtual void activate() = 0;
	virtual void deactivate() = 0;

	virtual void setup(const TF::Size index) = 0;
	virtual void setHistogram(TF::Histogram::Ptr histogram) = 0;
	virtual void setDomain(const TF::Size domain) = 0;

	virtual bool connectToTFPalette(QObject* tfPalette) = 0;	//	tfPalette has to be TFPalette instance
	virtual bool createPaletteButton(QWidget* parent) = 0;
	virtual void createDockWidget(QWidget* parent) = 0;

	virtual TF::Size getIndex() = 0;

	virtual TFPaletteButton* getButton() const = 0;
	virtual QDockWidget* getDockWidget() const = 0;

	virtual bool changed() = 0;

	template<typename BufferIterator>
	bool applyTransferFunction(
		BufferIterator begin,
		BufferIterator end){

		return TF::Adaptation::applyTransferFunction<BufferIterator>(begin, end, functionToApply_());
	}

protected:
	
	TFHolderInterface(){}

	virtual TFApplyFunctionInterface::Ptr functionToApply_() = 0;
};

} // namespace GUI
} // namespace M4D

#endif //TF_HOLDERINTERFACE
