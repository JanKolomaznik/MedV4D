#ifndef TF_COMPOSITE_MODIFIER
#define TF_COMPOSITE_MODIFIER

#include <TFModifier1D.h>
#include <TFCompositionDialog.h>

#include <QtCore/QTimer>
#include <QtGui/QVBoxLayout>

#include <ui_TFCompositeModifier.h>

namespace M4D {
namespace GUI {

class TFPalette;

class TFCompositeModifier: public TFModifier1D{

	Q_OBJECT

public:

	typedef boost::shared_ptr<TFCompositeModifier> Ptr;

	TFCompositeModifier(
		TFFunctionInterface::Ptr function,
		TFPainter1D::Ptr painter,		
		TFPalette* palette);

	~TFCompositeModifier();

protected slots:

	void manageComposition_clicked();
	void change_check();
	void changeChecker_intervalChange(int value);

protected:

	struct Editor{
		TFEditor* editor;
		Common::TimeStamp change;
		QLabel* name;

		void updateName();

		Editor(TFEditor* editor);

		~Editor();
	};
	typedef std::map<TF::Size, Editor*> Composition;

	typedef TFCompositionDialog::Selection Selection;

	Ui::TFCompositeModifier* compositeTools_;
	QWidget* compositeWidget_;
	QVBoxLayout* layout_;
	QSpacerItem* pushUpSpacer_;
	
	TFPalette* palette_;
	Common::TimeStamp lastPaletteChange_;
	std::map<TF::Size, TFEditor*> editors_;

	TFCompositionDialog manager_;
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