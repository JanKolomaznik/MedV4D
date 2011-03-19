#ifndef TF_CREATOR
#define TF_CREATOR

#include <QtGui/QDialog>
#include <QtGui/QVBoxLayout>

#include <TFCommon.h>
#include <TFHolder.h>

#include <TFDialogButtons.h>

#include <TFFunctions.h>
#include <TFPainters.h>
#include <TFModifiers.h>
#include <TFPredefined.h>

#include "ui_TFCreator.h"

namespace M4D {
namespace GUI {

class TFCreator: public QDialog{

	Q_OBJECT

public:

	static TFHolder* createTransferFunction(QMainWindow* mainWindow, const TF::Size domain);

	static TFHolder* loadTransferFunction(QMainWindow* mainWindow, const TF::Size domain);

private slots:

	void on_nextButton_clicked();
	void on_backButton_clicked();

	void predefinedButton_clicked(TF::Types::Predefined predefined);
	void functionButton_clicked(TF::Types::Function function);
	void painterButton_clicked(TF::Types::Painter painter);
	void modifierButton_clicked(TF::Types::Modifier modifier);

private:	

	TFCreator(QWidget* parent = 0);
	~TFCreator();

	TF::Types::PredefinedStructure getResult();

	enum State{
		Predefined,
		Function,
		Painter,
		Modifier
	};
	State state_;

	Ui::TFCreator* ui_;

	QVBoxLayout* predefinedLayout_;
	QVBoxLayout* functionLayout_;
	QVBoxLayout* otherLayout_;

	TF::Types::PredefinedStructure structure_;

	bool predefinedSet_;
	bool functionSet_;
	bool painterSet_;
	bool modifierSet_;

	void setStatePredefined_();
	void setStateFunction_();
	void setStatePainter_();
	void setStateModifier_();	

};

} // namespace GUI
} // namespace M4D

#endif	//TF_CREATOR