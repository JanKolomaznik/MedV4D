#ifndef TF_COMPOSITE_MODIFIER
#define TF_COMPOSITE_MODIFIER

#include "MedV4D/GUI/TF/Modifier1D.h"
#include "MedV4D/GUI/TF/CompositionDialog.h"

#include <QtCore/QTimer>
#include <QtWidgets/QVBoxLayout>

#include "MedV4D/generated/ui_CompositeModifier.h"

namespace M4D {
namespace GUI {

class Palette;

class CompositeModifier: public Modifier1D{

	Q_OBJECT

public:

	typedef std::shared_ptr<CompositeModifier> Ptr;

	CompositeModifier(
		TransferFunctionInterface::Ptr function,
		Painter1D::Ptr painter,
		Palette* palette);

	~CompositeModifier();

protected slots:

	void manageComposition_clicked();
	void change_check();
	void changeChecker_intervalChange(int value);

protected:

	struct EditorInstance {
		Editor* editor;
		Common::TimeStamp change;
		QLabel* name;

		void updateName();

		EditorInstance(Editor* editor);

		~EditorInstance();
	};
	typedef std::map<TF::Size, EditorInstance*> Composition;

	typedef CompositionDialog::Selection Selection;

	Ui::CompositeModifier* compositeTools_;
	QWidget* compositeWidget_;
	QVBoxLayout* layout_;
	QSpacerItem* pushUpSpacer_;

	Palette* palette_;
	Common::TimeStamp lastPaletteChange_;
	std::map<TF::Size, Editor*> editors_;

	CompositionDialog manager_;
	bool managing_;

	QTimer changeChecker_;
	Composition composition_;

	virtual void mousePressEvent(QMouseEvent *e);

	virtual void computeResultFunction_();
	void updateComposition_();

	virtual void createTools_();
	void clearLayout_();
};

} // namespace GUI
} // namespace M4D

#endif //TF_COMPOSITE_MODIFIER
