#include "MedV4D/GUI/TF/CompositeModifier.h"

#include "MedV4D/GUI/TF/Palette.h"
#include "MedV4D/GUI/TF/Editor.h"

#include <QtWidgets/QMessageBox>
#include "MedV4D/Common/Common.h"
#include "MedV4D/Common/Debug.h"

namespace M4D {
namespace GUI {

CompositeModifier::CompositeModifier(
		TransferFunctionInterface::Ptr function,
		Painter1D::Ptr painter,
		Palette* palette
		):
	Modifier1D(function, painter),
	palette_(palette),
	editors_(palette->getEditors()),
	managing_(false)
{
	//D_BLOCK_COMMENT( TO_STRING(__FUNCTION__ << " entered"), TO_STRING(__FUNCTION__ << " leaved") );
	compositeTools_ = new Ui::CompositeModifier;
	compositeWidget_ = new QWidget;
	layout_ = new QVBoxLayout;
	pushUpSpacer_ = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

	compositeTools_->setupUi(compositeWidget_);

	bool compositionEnabled = false;
	for(Palette::Editors::iterator it = editors_.begin(); it != editors_.end(); ++it)
	{
		if(!it->second->hasAttribute(Editor::Composition) &&
			it->second->getDimension() == TF_DIMENSION_1)
		{
			compositionEnabled = true;
			break;
		}
	}
	compositeTools_->manageButton->setEnabled(compositionEnabled);

	layout_->setContentsMargins(10,10,10,10);
	compositeTools_->scrollAreaWidget->setLayout(layout_);

	bool manageConnected = QObject::connect(compositeTools_->manageButton, SIGNAL(clicked()),
		this, SLOT(manageComposition_clicked()));
	tfAssert(manageConnected);

	bool delaySpinConnected = QObject::connect(compositeTools_->delaySpin, SIGNAL(valueChanged(int)),
		this, SLOT(changeChecker_intervalChange(int)));
	tfAssert(delaySpinConnected);

	bool timerConnected = QObject::connect(&changeChecker_, SIGNAL(timeout()), this, SLOT(change_check()));
	tfAssert(timerConnected);
	changeChecker_.setInterval(compositeTools_->delaySpin->value());
	changeChecker_.start();

	activeView_ = ActiveAlpha;
}

CompositeModifier::~CompositeModifier(){

	delete compositeTools_;
}

void CompositeModifier::createTools_(){

	simpleWidget_->hide();

    QFrame* separator = new QFrame();
    separator->setFrameShape(QFrame::HLine);
    separator->setFrameShadow(QFrame::Sunken);

	QVBoxLayout* layout = new QVBoxLayout();
	layout->addItem(centerWidget_(compositeWidget_));
	layout->addWidget(separator);
	layout->addItem(centerWidget_(viewWidget_));

	toolsWidget_ = new QWidget();
	toolsWidget_->setLayout(layout);
}

void CompositeModifier::mousePressEvent(QMouseEvent *e){

	TF::PaintingPoint relativePoint = getRelativePoint_(e->x(), e->y());
	if(relativePoint == ignorePoint_) return;

	if(e->button() == Qt::LeftButton)
	{
		leftMousePressed_ = true;
		inputHelper_ = relativePoint;
	}

	ViewModifier::mousePressEvent(e);
}

void CompositeModifier::changeChecker_intervalChange(int value){

	changeChecker_.setInterval(value);
}

void CompositeModifier::clearLayout_(){

	layout_->removeItem(pushUpSpacer_);
	QLayoutItem* layoutIt;
	while(!layout_->isEmpty())
	{
		layoutIt = layout_->itemAt(0);
		layout_->removeItem(layoutIt);
	}
}

void CompositeModifier::manageComposition_clicked(){

	change_check();
	if(!compositeTools_->manageButton->isEnabled()) return;

	manager_.updateSelection(editors_, palette_);

	managing_ = true;
	manager_.exec();
	managing_ = false;

	updateComposition_();
}

void CompositeModifier::updateComposition_(){

	Selection selection = manager_.getComposition();
	clearLayout_();

	bool recalculate = false;
	M4D::Common::TimeStamp lastChange;
	EditorInstance* editor = nullptr;
	Composition newComposition;
	Composition::iterator found;
	for(Selection::iterator it = selection.begin(); it != selection.end(); ++it)
	{
		found = composition_.find(*it);
		if(found == composition_.end())
		{
			editor = new EditorInstance(editors_.find(*it)->second);
			newComposition.insert(std::make_pair(
				*it,
				editor)
			);
			recalculate = true;
		}
		else
		{
			lastChange = found->second->editor->lastChange();
			if(found->second->change != lastChange)
			{
				recalculate = true;
				found->second->change = lastChange;
			}
			editor = found->second;
			editor->updateName();

			newComposition.insert(*found);
			composition_.erase(found);
		}

		layout_->addWidget(editor->name);
	}
	layout_->addItem(pushUpSpacer_);

	if(!composition_.empty()) recalculate = true;
	for(Composition::iterator it = composition_.begin(); it != composition_.end(); ++it)
	{
		layout_->removeWidget(it->second->name);
		delete it->second;
	}
	composition_.swap(newComposition);

	if(recalculate) computeResultFunction_();
}

void CompositeModifier::change_check(){

	if(managing_)
	{
		updateComposition_();
		return;
	}

	bool recalculate = false;

	Common::TimeStamp lastChange = palette_->lastPaletteChange();
	if(lastPaletteChange_ != lastChange)
	{
		lastPaletteChange_ = lastChange;
		editors_ = palette_->getEditors();
		//editors_.swap(palette_->getEditors());

		bool compositionEnabled = false;
		Composition newComposition;
		Composition::iterator found;
		for(Palette::Editors::iterator it = editors_.begin(); it != editors_.end(); ++it)
		{
			if(!it->second->hasAttribute(Editor::Composition) &&
				it->second->getDimension() == TF_DIMENSION_1)
			{
				compositionEnabled = true;

				found = composition_.find(it->first);
				if(found != composition_.end())
				{
					newComposition.insert(*found);
					composition_.erase(found);
				}
			}
		}

		if(!composition_.empty()) recalculate = true;
		for(Composition::iterator it = composition_.begin(); it != composition_.end(); ++it)
		{
			layout_->removeWidget(it->second->name);
			delete it->second;
		}
		composition_.swap(newComposition);

		compositeTools_->manageButton->setEnabled(compositionEnabled);
	}

	for(Composition::iterator it = composition_.begin(); it != composition_.end(); ++it)
	{
		lastChange = it->second->editor->lastChange();
		if(it->second->change != lastChange)
		{
			recalculate = true;
			it->second->change = lastChange;
		}
		it->second->updateName();
	}

	if(recalculate) computeResultFunction_();
}

void CompositeModifier::computeResultFunction_(){

	TransferFunctionInterface::Ptr function = workCopy_->getFunction();
	TF::Size domain = function->getDomain(TF_DIMENSION_1);
	for(Composition::iterator it = composition_.begin(); it != composition_.end(); ++it)
	{	//check if dimension change is in process
		if(it->second->editor->getFunction().getDomain(TF_DIMENSION_1) != domain) return;
	}

	for(TF::Size i = 0; i < domain; ++i)
	{
		coords_[0] = i;
		TF::Color result;
		TF::Color color;
		for(Composition::iterator it = composition_.begin(); it != composition_.end(); ++it)
		{
			color = it->second->editor->getFunction().getRGBfColor(coords_);
			result.component1 += (color.component1*color.alpha);
			result.component2 += (color.component2*color.alpha);
			result.component3 += (color.component3*color.alpha);
		}

		if(result.component1 > 1) result.component1 = 1;
		if(result.component2 > 1) result.component2 = 1;
		if(result.component3 > 1) result.component3 = 1;
		result.alpha = function->getRGBfColor(coords_).alpha;

		function->setRGBfColor(coords_, result);
	}
	workCopy_->forceUpdate();
	++stamp_;
	changed_ = true;
	update();
}

//---CompositeModifier::EditorInstance---

void CompositeModifier::EditorInstance::updateName(){

	QString newName = QString::fromStdString(editor->getName());
	if(name->text() != newName) name->setText(newName);
}

CompositeModifier::EditorInstance::EditorInstance(Editor* editor):
	editor(editor),
	change(editor->lastChange()),
	name(new QLabel(QString::fromStdString(editor->getName())))
{
}

CompositeModifier::EditorInstance::~EditorInstance(){

	name->hide();
	delete name;
}

} // namespace GUI
} // namespace M4D
