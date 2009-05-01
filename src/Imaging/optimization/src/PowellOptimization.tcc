/**
 * Based on the powell algorithm source code, presented in the Numerical Recipes in C++ book.
 */
#ifndef POWELL_OPTIMIZATION_H
#error PowellOptimization.tcc cannot be included directly!
#else

namespace M4D
{
namespace Imaging
{

template< typename ElementType, uint32 dim >
void
PowellOptimization< ElementType, dim >
::optimize(Vector< ElementType, dim > &v, ElementType &fret, ElementType func(Vector< ElementType, dim > &))
{
	extern_functor = func;
        const ElementType FTOL=1.0e-6;
        ElementType p_d[dim];
        int i,j,iter;
	for (i=0;i<dim;i++) p_d[i] = v[i];
        NRVec< ElementType > p(p_d,dim);
        NRMat< ElementType > xi(dim,dim);

        for (i=0;i<dim;i++)
          for (j=0;j<dim;j++)
            xi[i][j]=(i == j ? 1.0 : 0.0);
        powell(p,xi,FTOL,iter,fret,&M4D::Imaging::PowellOptimization< ElementType, dim >::caller);
	for (i=0;i<dim;i++) v[i] = p[i];
}

template< typename ElementType, uint32 dim >
void
PowellOptimization< ElementType, dim >
::powell(NRVec< ElementType > &p, NRMat< ElementType > &xi, const ElementType ftol, int &iter,
        ElementType &fret, VectorFunc func)
{
        const int ITMAX=200;
        const ElementType TINY=1.0e-25;
        int i,j,ibig;
        ElementType del,fp,fptt,t;

        int n=p.size();
        NRVec< ElementType > pt(n),ptt(n),xit(n);
        fret=(this->*func)(p);
        for (j=0;j<n;j++) pt[j]=p[j];
        for (iter=0;;++iter) {
                fp=fret;
                ibig=0;
                del=0.0;
                for (i=0;i<n;i++) {
                        for (j=0;j<n;j++) xit[j]=xi[j][i];
                        fptt=fret;
                        linmin(p,xit,fret,func);
                        if (fptt-fret > del) {
                                del=fptt-fret;
                                ibig=i+1;
                        }
                }
                if (2.0*(fp-fret) <= ftol*(fabs(fp)+fabs(fret))+TINY) {
                        return;
                }
                if (iter == ITMAX) NR::nrerror("powell exceeding maximum iterations.");
                for (j=0;j<n;j++) {
                        ptt[j]=2.0*p[j]-pt[j];
                        xit[j]=p[j]-pt[j];
                        pt[j]=p[j];
                }
                fptt=(this->*func)(ptt);
                if (fptt < fp) {
                        t=2.0*(fp-2.0*fret+fptt)*SQR(fp-fret-del)-del*SQR(fp-fptt);
                        if (t < 0.0) {
                                linmin(p,xit,fret,func);
                                for (j=0;j<n;j++) {
                                        xi[j][ibig-1]=xi[j][n-1];
                                        xi[j][n-1]=xit[j];
                                }
                        }
                }
        }
}

template< typename ElementType, uint32 dim >
void
PowellOptimization< ElementType, dim >
::linmin(NRVec< ElementType > &p, NRVec< ElementType > &xi, ElementType &fret, VectorFunc func)
{
        int j;
        const ElementType TOL=1.0e-8;
        ElementType xx,xmin,fx,fb,fa,bx,ax;

        int n=p.size();
        ncom=n;
        pcom_p=new NRVec< ElementType >(n);
        xicom_p=new NRVec< ElementType >(n);
        nrfunc=func;
        NRVec< ElementType > &pcom=*pcom_p,&xicom=*xicom_p;
        for (j=0;j<n;j++) {
                pcom[j]=p[j];
                xicom[j]=xi[j];
        }
        ax=0.0;
        xx=1.0;
        mnbrak(ax,xx,bx,fa,fx,fb,&M4D::Imaging::PowellOptimization< ElementType, dim >::f1dim);
        fret=brent(ax,xx,bx,&M4D::Imaging::PowellOptimization< ElementType, dim >::f1dim,TOL,xmin);
        for (j=0;j<n;j++) {
                xi[j] *= xmin;
                p[j] += xi[j];
        }
        delete xicom_p;
        delete pcom_p;
}

template< typename ElementType, uint32 dim >
void
PowellOptimization< ElementType, dim >
::mnbrak(ElementType &ax, ElementType &bx, ElementType &cx, ElementType &fa, ElementType &fb, ElementType &fc,
        SingleFunc func)
{
        const ElementType GOLD=1.618034,GLIMIT=100.0,TINY=1.0e-20;
        ElementType ulim,u,r,q,fu;

        fa=(this->*func)(ax);
        fb=(this->*func)(bx);
        if (fb > fa) {
                SWAP(ax,bx);
                SWAP(fb,fa);
        }
        cx=bx+GOLD*(bx-ax);
        fc=(this->*func)(cx);
        while (fb > fc) {
                r=(bx-ax)*(fb-fc);
                q=(bx-cx)*(fb-fa);
                u=bx-((bx-cx)*q-(bx-ax)*r)/
                        (2.0*SIGN(MAX(fabs(q-r),TINY),q-r));
                ulim=bx+GLIMIT*(cx-bx);
                if ((bx-u)*(u-cx) > 0.0) {
                        fu=(this->*func)(u);
                        if (fu < fc) {
                                ax=bx;
                                bx=u;
                                fa=fb;
                                fb=fu;
                                return;
                        } else if (fu > fb) {
                                cx=u;
                                fc=fu;
                                return;
                        }
                        u=cx+GOLD*(cx-bx);
                        fu=(this->*func)(u);
                } else if ((cx-u)*(u-ulim) > 0.0) {
                        fu=(this->*func)(u);
                        if (fu < fc) {
                                shft3(bx,cx,u,u+GOLD*(u-cx));
                                shft3(fb,fc,fu,(this->*func)(u));
                        }
                } else if ((u-ulim)*(ulim-cx) >= 0.0) {
                        u=ulim;
                        fu=(this->*func)(u);
                } else {
                        u=cx+GOLD*(cx-bx);
                        fu=(this->*func)(u);
                }
                shft3(ax,bx,cx,u);
                shft3(fa,fb,fc,fu);
        }
}

template< typename ElementType, uint32 dim >
ElementType
PowellOptimization< ElementType, dim >
::f1dim(const ElementType x)
{
        int j;

        NRVec< ElementType > xt(ncom);
        NRVec< ElementType > &pcom=*pcom_p,&xicom=*xicom_p;
        for (j=0;j<ncom;j++)
                xt[j]=pcom[j]+x*xicom[j];
        return (this->*nrfunc)(xt);
}


template< typename ElementType, uint32 dim >
ElementType
PowellOptimization< ElementType, dim >
::brent(const ElementType ax, const ElementType bx, const ElementType cx, SingleFunc f,
        const ElementType tol, ElementType &xmin)
{
        const int ITMAX=100;
        const ElementType CGOLD=0.3819660;
        const ElementType ZEPS=numeric_limits<ElementType>::epsilon()*1.0e-3;
        int iter;
        ElementType a,b,d=0.0,etemp,fu,fv,fw,fx;
        ElementType p,q,r,tol1,tol2,u,v,w,x,xm;
        ElementType e=0.0;

        a=(ax < cx ? ax : cx);
        b=(ax > cx ? ax : cx);
        x=w=v=bx;
        fw=fv=fx=(this->*f)(x);
        for (iter=0;iter<ITMAX;iter++) {
                xm=0.5*(a+b);
                tol2=2.0*(tol1=tol*fabs(x)+ZEPS);
                if (fabs(x-xm) <= (tol2-0.5*(b-a))) {
                        xmin=x;
                        return fx;
                }
                if (fabs(e) > tol1) {
                        r=(x-w)*(fx-fv);
                        q=(x-v)*(fx-fw);
                        p=(x-v)*q-(x-w)*r;
                        q=2.0*(q-r);
                        if (q > 0.0) p = -p;
                        q=fabs(q);
                        etemp=e;
                        e=d;
                        if (fabs(p) >= fabs(0.5*q*etemp) || p <= q*(a-x) || p >= q*(b-x))
                                d=CGOLD*(e=(x >= xm ? a-x : b-x));
                        else {
                                d=p/q;
                                u=x+d;
                                if (u-a < tol2 || b-u < tol2)
                                        d=SIGN(tol1,xm-x);
                        }
                } else {
                        d=CGOLD*(e=(x >= xm ? a-x : b-x));
                }
                u=(fabs(d) >= tol1 ? x+d : x+SIGN(tol1,d));
                fu=(this->*f)(u);
                if (fu <= fx) {
                        if (u >= x) a=x; else b=x;
                        shft3(v,w,x,u);
                        shft3(fv,fw,fx,fu);
                } else {
                        if (u < x) a=u; else b=u;
                        if (fu <= fw || w == x) {
                                v=w;
                                w=u;
                                fv=fw;
                                fw=fu;
                        } else if (fu <= fv || v == x || v == w) {
                                v=u;
                                fv=fu;
                        }
                }
        }
	NR::nrerror("Too many iterations in brent");
        xmin=x;
        return fx;
}

} /*namespace Imaging*/
} /*namespace M4D*/

#endif
