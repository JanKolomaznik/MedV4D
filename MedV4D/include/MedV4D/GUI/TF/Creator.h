#ifndef TF_CREATOR
#define TF_CREATOR

#include <QtWidgets/QDialog>
#include <QtWidgets/QVBoxLayout>

#include "MedV4D/GUI/TF/Common.h"

#include "MedV4D/GUI/TF/CreatorDialogButtons.h"

#include "MedV4D/GUI/TF/AbstractFunction.h"
#include "MedV4D/GUI/TF/AbstractModifier.h"
#include "MedV4D/GUI/TF/AbstractPainter.h"
#include "MedV4D/GUI/TF/WorkCopy.h"

#include "MedV4D/GUI/TF/Dimensions.h"
#include "MedV4D/GUI/TF/Functions.h"
#include "MedV4D/GUI/TF/Painters.h"
#include "MedV4D/GUI/TF/Modifiers.h"
#include "MedV4D/GUI/TF/Predefined.h"

#include "MedV4D/GUI/TF/Editor.h"

#include "MedV4D/generated/ui_Creator.h"

namespace M4D {
namespace GUI {

class Palette;

class Creator: public QDialog{

	Q_OBJECT

public:

	Creator(QMainWindow* mainWindow, Palette* palette, const std::vector<TF::Size>& dataStructure);
	~Creator();

	Editor*
	createEditor();

	Editor*
	loadEditorFromFile( QString fileName );

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

	Ui::Creator* ui_;
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
	Palette* palette_;
	std::vector<TF::Size> dataStructure_;

	void setStateModeSelection_();
	void setStatePredefined_();
	void setStateDimension_();
	void setStateFunction_();
	void setStatePainter_();
	void setStateModifier_();

	void clearLayout_(bool deleteItems = true);

	Editor* loadEditor_();
	Editor* load_(TF::XmlReaderInterface* reader, bool& sideError);
	Editor* createEditor_();

	template<TF::Size dim>
	AbstractFunction<dim>* createFunction_();
	AbstractPainter* createPainter_(Editor::Attributes& attributes);
	AbstractModifier* createModifier_(Editor::Attributes& attributes);
};

} // namespace GUI
} // namespace M4D

#endif	//TF_CREATOR
