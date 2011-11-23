#ifndef TF_CREATOR
#define TF_CREATOR

#include <QtGui/QDialog>
#include <QtGui/QVBoxLayout>

#include <TFCommon.h>

#include <TFCreatorDialogButtons.h>

#include <TFAbstractFunction.h>
#include <TFAbstractModifier.h>
#include <TFAbstractPainter.h>
#include <TFWorkCopy.h>

#include <TFDimensions.h>
#include <TFFunctions.h>
#include <TFPainters.h>
#include <TFModifiers.h>
#include <TFPredefined.h>

#include <TFEditor.h>

#include <ui_TFCreator.h>

namespace M4D {
namespace GUI {

class TFPalette;

class TFCreator: public QDialog{

	Q_OBJECT

public:

	TFCreator(QMainWindow* mainWindow, TFPalette* palette, const std::vector<TF::Size>& dataStructure);
	~TFCreator();

	TFEditor* createEditor();

	void setDataStructure(const std::vector<TF::Size>& dataStructure);

private slots:

	void on_nextButton_clicked();
	void on_backButton_clicked();

	void mode_clicked();
	void dimensionButton_clicked(TF::Types::Dimension dimension);
	void predefinedButton_clicked(TF::Types::Predefined predefined);
	void functionButton_clicked(TF::Types::Function function);
	void painterButton_clicked(TF::Types::Painter painter);
	void modifierButton_clicked(TF::Types::Modifier modifier);

private:	

	enum State{
		ModeSelection,
		Predefined,
		Dimension,
		Function,
		Painter,
		Modifier
	};
	enum Mode{
		CreateCustom = 0,
		CreatePredefined = 1,
		CreateLoaded = 2
	};

	Ui::TFCreator* ui_;
	QVBoxLayout* layout_;
	QRadioButton* predefinedRadio_;
	QRadioButton* customRadio_;
	QRadioButton* loadRadio_;

	TF::XmlReaderInterface* reader_;

	State state_;
	Mode mode_;

	TF::Types::Structure structure_[3];
	std::string name_;

	bool predefinedSet_;
	bool dimensionSet_;
	bool functionSet_;
	bool painterSet_;
	bool modifierSet_;
	
	QMainWindow* mainWindow_;
	TFPalette* palette_;
	std::vector<TF::Size> dataStructure_;

	void setStateModeSelection_();
	void setStatePredefined_();
	void setStateDimension_();
	void setStateFunction_();
	void setStatePainter_();
	void setStateModifier_();	

	void clearLayout_(bool deleteItems = true);

	TFEditor* loadEditor_();
	TFEditor* load_(TF::XmlReaderInterface* reader, bool& sideError);
	TFEditor* createEditor_();

	template<TF::Size dim>
	typename TFAbstractFunction<dim>* createFunction_();
	TFAbstractPainter* createPainter_(TFEditor::Attributes& attributes);
	TFAbstractModifier* createModifier_(TFEditor::Attributes& attributes);
};

} // namespace GUI
} // namespace M4D

#endif	//TF_CREATOR