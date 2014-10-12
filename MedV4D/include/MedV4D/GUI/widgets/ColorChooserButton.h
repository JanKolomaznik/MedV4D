#ifndef COLOR_CHOOSER_BUTTON_H
#define COLOR_CHOOSER_BUTTON_H

#include <QtWidgets>

class ColorChooserButton: public QPushButton
{
	Q_OBJECT;
public:
	ColorChooserButton( QWidget *parent = NULL ): QPushButton( parent ), mColor( 255,255,255,255 ), mAllowAlphaSetting( false )
	{
		mColorDialog = new QColorDialog( this );
		QObject::connect( this, SIGNAL( clicked() ), this, SLOT( showDialog() ) );
		QObject::connect( mColorDialog, SIGNAL( colorSelected( const QColor & ) ), this, SLOT( setColor( const QColor & ) ) );
	}

	QColor
	color()const
	{
		return mColor;
	}

	bool
	isAlphaEnabled()const
	{
		return mAllowAlphaSetting;
	}

	QColor getIdealTextColor(const QColor& aBackgroundColor) const
	{
		const int cThreshold = 105;
		int backgroundDelta = (aBackgroundColor.red() * 0.299) + (aBackgroundColor.green() * 0.587) + (aBackgroundColor.blue() * 0.114);
		return QColor((255- backgroundDelta < cThreshold) ? Qt::black : Qt::white);
	}
public slots:
	void
	setColor( const QColor &aColor ) {
		mColor = aColor;

		static const QString cColorStyle("QPushButton { background-color : %1; color : %2; }");
		QColor idealTextColor = getIdealTextColor(aColor);
		setStyleSheet(cColorStyle.arg(aColor.name()).arg(idealTextColor.name()));

		update();
		emit colorUpdated();
	}

	void
	enableAlpha( bool aEnable )
	{
		mAllowAlphaSetting = aEnable;
	}
signals:
	void
	colorUpdated();

protected slots:
	void
	showDialog()
	{
		mColorDialog->setOption( QColorDialog::ShowAlphaChannel, mAllowAlphaSetting );
		mColorDialog->setCurrentColor( mColor );
		mColorDialog->exec();
		setColor(mColorDialog->currentColor());
		update();
	}

protected:
	/*void
	paintEvent ( QPaintEvent *event )
	{
		//QPushButton::paintEvent ( event );

		QPainter painter(this);
		QRect rect = QRect( QPoint(), size() );
		//TODO
		painter.setBrush( QBrush( mColor ) );
		painter.drawRect(rect);
	}*/


	QColor mColor;
	bool mAllowAlphaSetting;
	QColorDialog *mColorDialog;

};


#endif /*COLOR_CHOOSER_BUTTON_H*/
