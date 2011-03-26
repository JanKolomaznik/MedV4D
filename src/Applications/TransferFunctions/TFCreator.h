#ifndef TF_CREATOR
#define TF_CREATOR

#include <QtGui/QDialog>
#include <QtGui/QVBoxLayout>

#include <TFCommon.h>

#include <TFDialogButtons.h>

#include <TFAbstractFunction.h>
#include <TFAbstractModifier.h>
#include <TFAbstractPainter.h>
#include <TFWorkCopy.h>

#include <TFHolders.h>
#include <TFFunctions.h>
#include <TFPainters.h>
#include <TFModifiers.h>
#include <TFPredefined.h>

#include <TFHolderInterface.h>

#include <ui_TFCreator.h>

namespace M4D {
namespace GUI {

class TFCreator: public QDialog{

	Q_OBJECT

public:

	TFCreator(QMainWindow* mainWindow, const TF::Size domain);
	~TFCreator();

	TFHolderInterface* createTransferFunction();
	TFHolderInterface* loadTransferFunction();

	void setDomain(const TF::Size domain);

private slots:

	void on_nextButton_clicked();
	void on_backButton_clicked();

	void holderButton_clicked(TF::Types::Holder holder);
	void predefinedButton_clicked(TF::Types::Predefined predefined);
	void functionButton_clicked(TF::Types::Function function);
	void painterButton_clicked(TF::Types::Painter painter);
	void modifierButton_clicked(TF::Types::Modifier modifier);

private:	

	enum State{
		Holder,
		Predefined,
		Function,
		Painter,
		Modifier
	};
	State state_;

	Ui::TFCreator* ui_;

	QVBoxLayout* layout_;

	bool predefinedChoice_;
	TF::Types::Structure customStructure_;
	TF::Types::Structure predefinedStructure_;

	bool holderSet_;
	bool predefinedSet_;
	bool functionSet_;
	bool painterSet_;
	bool modifierSet_;
	
	QMainWindow* mainWindow_;
	TF::Size domain_;

	TF::Types::Structure& getStructure_();

	void setStateHolder_();
	void setStatePredefined_();
	void setStateFunction_();
	void setStatePainter_();
	void setStateModifier_();	

	void clearLayout_();

	TFHolderInterface* createHolder_();

	template<TF::Size dim>
	typename TFAbstractFunction<dim>::Ptr createFunction_();

	template<TF::Size dim>
	typename TFAbstractPainter<dim>::Ptr createPainter_();

	template<TF::Size dim>
	typename TFAbstractModifier<dim>::Ptr createModifier_(typename TFWorkCopy<dim>::Ptr workCopy);
};

} // namespace GUI
} // namespace M4D

#endif	//TF_CREATOR