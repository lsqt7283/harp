# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 18:55:36 2022

Hierarchical Clustering Asset Allocation

Adapted from:
    David Bailey & Marcos Lopez de Prado (2013) "An Open-Source Implementation of the Critical-Line Algorithm for Portfolio Optimization"
    Marcos Lopez de Prado (2016) "Building diversified portfolios that outperform out of sample"
    Thomas Raffinot (2017) "Hierarchical Clustering based Asset Allocation"
    Berowne Hlavaty & Robert Smith (2017) "Post-Modern Portfolio Construction: Examining Recent Innovations in Asset Allocation"

@author: TQiu
"""


import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd


class CLA:
    def __init__(self,mean,covar,lB,uB):
        # Initialize the class
        if (mean==np.ones(mean.shape)*mean.mean()).all():mean[-1,0]+=1e-5
        self.mean=mean
        self.covar=covar
        self.lB=lB
        self.uB=uB
        self.w=[] # solution
        self.l=[] # lambdas
        self.g=[] # gammas
        self.f=[] # free weights
#---------------------------------------------------------------
    def solve(self):
        # Compute the turning points,free sets and weights
        f,w=self.initAlgo()
        self.w.append(np.copy(w)) # store solution
        self.l.append(None)
        self.g.append(None)
        self.f.append(f[:])
        while True:
            #1) case a): Bound one free weight
            l_in=None
            if len(f)>1:
                covarF,covarFB,meanF,wB=self.getMatrices(f)
                covarF_inv=np.linalg.inv(covarF)
                j=0
                for i in f:
                    l,bi=self.computeLambda(covarF_inv,covarFB,meanF,wB,j,[self.lB[i],self.uB[i]])
                    if l>l_in:l_in,i_in,bi_in=l,i,bi
                    j+=1
            #2) case b): Free one bounded weight
            l_out=None
            if len(f)<self.mean.shape[0]:
                b=self.getB(f)
                for i in b:
                    covarF,covarFB,meanF,wB=self.getMatrices(f+[i])
                    covarF_inv=np.linalg.inv(covarF)
                    l,bi=self.computeLambda(covarF_inv,covarFB,meanF,wB,meanF.shape[0]-1, \
                    self.w[-1][i])
                    if (self.l[-1]==None or l<self.l[-1]) and l>l_out:l_out,i_out=l,i
            if (l_in==None or l_in<0) and (l_out==None or l_out<0):
                #3) compute minimum variance solution
                self.l.append(0)
                covarF,covarFB,meanF,wB=self.getMatrices(f)
                covarF_inv=np.linalg.inv(covarF)
                meanF=np.zeros(meanF.shape)
            else:
                #4) decide lambda
                if l_in>l_out:
                    self.l.append(l_in)
                    f.remove(i_in)
                    w[i_in]=bi_in # set value at the correct boundary
                else:
                    self.l.append(l_out)
                    f.append(i_out)
                covarF,covarFB,meanF,wB=self.getMatrices(f)
                covarF_inv=np.linalg.inv(covarF)
            #5) compute solution vector
            wF,g=self.computeW(covarF_inv,covarFB,meanF,wB)
            for i in range(len(f)):w[f[i]]=wF[i]
            self.w.append(np.copy(w)) # store solution
            self.g.append(g)
            self.f.append(f[:])
            if self.l[-1]==0:break
        #6) Purge turning points
        self.purgeNumErr(10e-10)
        self.purgeExcess()
#---------------------------------------------------------------
    def initAlgo(self):
        # Initialize the algo
        #1) Form structured array
        a=np.zeros((self.mean.shape[0]),dtype=[('id',int),('mu',float)])
        b=[self.mean[i][0] for i in range(self.mean.shape[0])] # dump array into list
        a[:]=zip(range(self.mean.shape[0]),b) # fill structured array
        #2) Sort structured array
        b=np.sort(a,order='mu')
        #3) First free weight
        i,w=b.shape[0],np.copy(self.lB)
        while sum(w)<1:
            i-=1
            w[b[i][0]]=self.uB[b[i][0]]
        w[b[i][0]]+=1-sum(w)
        return [b[i][0]],w
#---------------------------------------------------------------
    def computeBi(self,c,bi):
        if c>0:bi=bi[1][0]
        if c<0:bi=bi[0][0]
        return bi
#---------------------------------------------------------------
    def computeW(self,covarF_inv,covarFB,meanF,wB):
        #1) compute gamma
        onesF=np.ones(meanF.shape)
        g1=np.dot(np.dot(onesF.T,covarF_inv),meanF)
        g2=np.dot(np.dot(onesF.T,covarF_inv),onesF)
        if wB==None:
            g,w1=float(-self.l[-1]*g1/g2+1/g2),0
        else:
            onesB=np.ones(wB.shape)
            g3=np.dot(onesB.T,wB)
            g4=np.dot(covarF_inv,covarFB)
            w1=np.dot(g4,wB)
            g4=np.dot(onesF.T,w1)
            g=float(-self.l[-1]*g1/g2+(1-g3+g4)/g2)
        #2) compute weights
        w2=np.dot(covarF_inv,onesF)
        w3=np.dot(covarF_inv,meanF)
        return -w1+g*w2+self.l[-1]*w3,g
#---------------------------------------------------------------
    def computeLambda(self,covarF_inv,covarFB,meanF,wB,i,bi):
        #1) C
        onesF=np.ones(meanF.shape)
        c1=np.dot(np.dot(onesF.T,covarF_inv),onesF)
        c2=np.dot(covarF_inv,meanF)
        c3=np.dot(np.dot(onesF.T,covarF_inv),meanF)
        c4=np.dot(covarF_inv,onesF)
        c=-c1*c2[i]+c3*c4[i]
        if c==0:return None,None
        #2) bi
        if type(bi)==list:bi=self.computeBi(c,bi)
        #3) Lambda
        if wB==None:
            # All free assets
            return float((c4[i]-c1*bi)/c),bi
        else:
            onesB=np.ones(wB.shape)
            l1=np.dot(onesB.T,wB)
            l2=np.dot(covarF_inv,covarFB)
            l3=np.dot(l2,wB)
            l2=np.dot(onesF.T,l3)
            return float(((1-l1+l2)*c4[i]-c1*(bi+l3[i]))/c),bi
#---------------------------------------------------------------
    def getMatrices(self,f):
        # Slice covarF,covarFB,covarB,meanF,meanB,wF,wB
        covarF=self.reduceMatrix(self.covar,f,f)
        meanF=self.reduceMatrix(self.mean,f,[0])
        b=self.getB(f)
        covarFB=self.reduceMatrix(self.covar,f,b)
        wB=self.reduceMatrix(self.w[-1],b,[0])
        return covarF,covarFB,meanF,wB
#---------------------------------------------------------------
    def getB(self,f):
        return self.diffLists(range(self.mean.shape[0]),f)
#---------------------------------------------------------------
    def diffLists(self,list1,list2):
        return list(set(list1)-set(list2))
#---------------------------------------------------------------
    def reduceMatrix(self,matrix,listX,listY):
        # Reduce a matrix to the provided list of rows and columns
        if len(listX)==0 or len(listY)==0:return
        matrix_=matrix[:,listY[0]:listY[0]+1]
        for i in listY[1:]:
            a=matrix[:,i:i+1]
            matrix_=np.append(matrix_,a,1)
        matrix__=matrix_[listX[0]:listX[0]+1,:]
        for i in listX[1:]:
            a=matrix_[i:i+1,:]
            matrix__=np.append(matrix__,a,0)
        return matrix__
#---------------------------------------------------------------
    def purgeNumErr(self,tol):
        # Purge violations of inequality constraints (associated with ill-conditioned covar matrix)
        i=0
        while True:
            if i==len(self.w):break
            w=self.w[i]
            for j in range(w.shape[0]):
                if w[j]-self.lB[j]<-tol or w[j]-self.uB[j]>tol:
                    del self.w[i]
                    del self.l[i]
                    del self.g[i]
                    del self.f[i]
                    break
            i+=1
#---------------------------------------------------------------
    def purgeExcess(self):
        # Remove violations of the convex hull
        i,repeat=0,False
        while True:
            if repeat==False:i+=1
            if i==len(self.w)-1:break
            w=self.w[i]
            mu=np.dot(w.T,self.mean)[0,0]
            j,repeat=i+1,False
            while True:
                if j==len(self.w):break
                w=self.w[j]
                mu_=np.dot(w.T,self.mean)[0,0]
                if mu<mu_:
                    del self.w[i]
                    del self.l[i]
                    del self.g[i]
                    del self.f[i]
                    repeat=True
                    break
                else:
                    j+=1
#---------------------------------------------------------------
    def getMinVar(self):
        # Get the minimum variance solution
        var=[]
        for w in self.w:
            a=np.dot(np.dot(w.T,self.covar),w)
            var.append(a)
        return min(var)**.5,self.w[var.index(min(var))]
#---------------------------------------------------------------
    def getMaxSR(self):
        # Get the max Sharpe ratio portfolio
        #1) Compute the local max SR portfolio between any two neighbor turning points
        w_sr,sr=[],[]
        for i in range(len(self.w)-1):
            w0=np.copy(self.w[i])
            w1=np.copy(self.w[i+1])
            kargs={'minimum':False,'args':(w0,w1)}
            a,b=self.goldenSection(self.evalSR,0,1,**kargs)
            w_sr.append(a*w0+(1-a)*w1)
            sr.append(b)
        return max(sr),w_sr[sr.index(max(sr))]
#---------------------------------------------------------------
    def evalSR(self,a,w0,w1):
        # Evaluate SR of the portfolio within the convex combination
        w=a*w0+(1-a)*w1
        b=np.dot(w.T,self.mean)[0,0]
        c=np.dot(np.dot(w.T,self.covar),w)[0,0]**.5
        return b/c
#---------------------------------------------------------------
    def goldenSection(self,obj,a,b,**kargs):
        # Golden section method. Maximum if kargs['minimum']==False is passed
        from math import log,ceil
        tol,sign,args=1.0e-9,1,None
        if 'minimum' in kargs and kargs['minimum']==False:sign=-1
        if 'args' in kargs:args=kargs['args']
        numIter=int(ceil(-2.078087*log(tol/abs(b-a))))
        r=0.618033989
        c=1.0-r
        # Initialize
        x1=r*a+c*b;x2=c*a+r*b
        f1=sign*obj(x1,*args);f2=sign*obj(x2,*args)
        # Loop
        for i in range(numIter):
            if f1>f2:
                a=x1
                x1=x2;f1=f2
                x2=c*a+r*b;f2=sign*obj(x2,*args)
            else:
                b=x2
                x2=x1;f2=f1
                x1=r*a+c*b;f1=sign*obj(x1,*args)
        if f1<f2:return x1,sign*f1
        else:return x2,sign*f2
#---------------------------------------------------------------
    def efFrontier(self,points):
        # Get the efficient frontier
        mu,sigma,weights=[],[],[]
        a=np.linspace(0,1,points/len(self.w))[:-1] # remove the 1, to avoid duplications
        b=range(len(self.w)-1)
        for i in b:
            w0,w1=self.w[i],self.w[i+1]
            if i==b[-1]:a=np.linspace(0,1,points/len(self.w)) # include the 1 in the last iteration
            for j in a:
                w=w1*j+(1-j)*w0
                weights.append(np.copy(w))
                mu.append(np.dot(w.T,self.mean)[0,0])
                sigma.append(np.dot(np.dot(w.T,self.covar),w)[0,0]**.5)
        return mu,sigma,weights
#---------------------------------------------------------------



def getCLA(cov, **kargs):
    # Compute CLA's minimum variance portfolio
    mean = np.arange(cov.shape[0]).reshape(-1,1) # Not used by C portf
    lB = np.zeros(mean.shape)
    uB = np.zeros(mean.shape)
    cla = CLA(mean, cov, lB, uB)
    cla.solve()
    return cla.w[-1].flatten()


def getHRP(cov, corr):
    # Construct a hierarchical portfolio
    corr, cov = pd.DataFrame(corr), pd.DataFrame(cov)
    dist = correlDist(corr)
    link = sch.linkage(dist, 'single')
    sortIx = getQuasiDiag(link)
    sortIx = corr.index[sortIx].tolist() # recover labels
    hrp = getRecBipart(cov, sortIx)
    return hrp.sort_index()


def getIVP(cov, **kargs):
    # Compute the inverse-variance portfolio
    ivp = 1. / np.diag(cov)
    ivp /= ivp.sum()
    return ivp


def getClusterVar(cov, cItems):
    # Compute variance per cluster
    cov_ = cov.loc[cItems,cItems] # matrix slice
    w_ = getIVP(cov_).reshape(-1,1)
    cVar = np.dot(np.dot(w_.T,cov_),w_)[0,0]
    return cVar


def getQuasiDiag(link):
    # Sort clustered items by distance
    link = link.astype(int)
    sortIx = pd.Series([link[-1,0], link[-1,1]])
    numItems = link[-1,3] # number of original items
    while sortIx.max() >= numItems:
        sortIx.index = range(0, sortIx.shape[0]*2, 2) # make space
        df0 = sortIx[sortIx>=numItems] # find clusters
        i = df0.index
        j = df0.values - numItems
        sortIx[i] = link[j,0] # item 1
        df0 = pd.Series(link[j,1], index=i+1)
        sortIx = pd.concat([sortIx, df0]) ### @tqiu ### #sortIx = sortIx.append(df0) # item 2
        sortIx = sortIx.sort_index() # re-sort
        sortIx.index = range(sortIx.shape[0]) # re-index
    return sortIx.tolist()


def getRecBipart(cov, sortIx):
    # Compute HRP alloc
    w = pd.Series(1,index=sortIx)
    cItems = [sortIx] # initialize all items in one cluster
    while len(cItems) > 0:
        cItems = [i[j:k] for i in cItems 
                  for j,k in ((0,len(i)//2), (len(i)//2,len(i))) 
                  if len(i)>1] # bi-section
        # parse in pairs
        for i in range(0, len(cItems), 2):
            cItems0 = cItems[i] # cluster 1
            cItems1 = cItems[i+1] # cluster 2
            cVar0 = getClusterVar(cov, cItems0)
            cVar1 = getClusterVar(cov, cItems1)
            alpha = 1 - cVar0/(cVar0+cVar1)
            w[cItems0] *= alpha # weight 1
            w[cItems1] *= 1-alpha # weight 2
    return w


def getHARP(cov, sortIx, xReturns=None, riskaversion=0.5, minWt = 0.001, maxWt = 1.0):
    ##### JPM Oct 10 2017, Berowne Hlavaty & Robert Smith #####
    # Compute HAP, HRP and HARP asset allocation
    # xReturns Asset Returns must be rescaled from zero to 1.0! getxprtnnZeroOne(expReturns)
    # riskaversion = Lambda relative importance of risk vs expected returns
    # Based on de Prado’s “getRecBipart” function
    w=pd.Series(1.0/sortIx.__len__() ,index=sortIx)
    wts=pd.DataFrame(data=0,index=sortIx, columns=[0])
    cItems=[sortIx] # initialize all items in one cluster
    wti=0
    while len(cItems)>0:
        cItems=[i[j:k] for i in cItems for j,k in ((0,len(i)//2), (len(i)//2,len(i))) if len(i)>1] # bi-section
        for i in range(0,len(cItems),2): # parse in pairs i=0
            cItems0=cItems[i] # cluster 1
            cItems1=cItems[i+1] # cluster 2
            cVar0=getClusterVar(cov,cItems0) # Single variance number for portfolio 0
            cVar1=getClusterVar(cov,cItems1)
            if xReturns is None:
                alpha=1-cVar0/(cVar0+cVar1)
                w[cItems0]*=alpha # weight 1
                w[cItems1]*=1-alpha # weight 2
            else:
                eVar0 = xReturns[cItems0].values.mean() # Expected Returns of group 1
                eVar1 = xReturns[cItems1].values.mean() # Expected Returns of group 2
                alpha = 1. - cVar0 / (cVar0+cVar1) # HRPi Relative Variance - HRP 'alpha'
                xprtn = 0. + eVar0 / (eVar0+eVar1) # HAPi Expected Return Relative
                # High Risk Aversion results in more stock weight from variance vs. returns.
                w[cItems0] *= riskaversion*alpha + (xprtn*(1.-riskaversion))# HARPi Wt1
                w[cItems1] *= riskaversion*(1.-alpha) + (1.-xprtn)*((1.-riskaversion))#Wt2
        wts[wti] = w #store incremental weights for debugging
        wti +=1
        w[w<minWt] = 0.0
        w = w.clip(0.0, maxWt)
        w[w<maxWt] = w[w<maxWt] / sum(w) # fix rounding errors
    return w


def getClusters(corr = pd.DataFrame, method='ward', bPlot=False):
    ##### JPM Oct 10 2017, Berowne Hlavaty & Robert Smith #####
    # HCP
    corr = corr.replace([np.inf, -np.inf], np.nan).fillna(0, inplace=False)
    clusters = []
    w=pd.Series(1.0,index=corr.index)
    Z = sch.linkage(corr, method) # 'ward', 'single', etc...)
    # Combine raw leaf nodes with linkage matrix
    raw = pd.DataFrame(corr.index, index=corr.index, columns=['cItems0'])
    raw['cItems1'] = np.nan
    raw['Dist'] = 0
    raw['Count'] = 1
    link = pd.DataFrame(Z, columns=['cItems0','cItems1','Dist','Count'])
    linked = pd.concat([raw, link],axis=0, ignore_index=True)
    # Initial Weights
    linked['wt']=1.
    # Traverse linkage in reverse order
    w = linked.__len__()-1
    while w>Z.__len__():
        inWt = linked.loc[w,'wt']/2.0 # weight split
        cItems0, cItems1 = linked.iloc[w,0:2].astype(int)
        linked.loc[[cItems0, cItems1],'wt'] *= inWt
        w-=1
    if bPlot: # Do we need a chart
        fig = PlotDendo(Z, labels=linked.loc[0:Z.__len__(),'wt'].values) # corr.index)
    return linked.loc[0:Z.__len__(),'wt']


def PlotDendo(Z, labels):
    ##### JPM Oct 10 2017, Berowne Hlavaty & Robert Smith #####
    plt.figure(figsize=(8, 8))
    plt.title('Hierarchical Clustering Dendrogram on Correlations')
    # plt.xlabel('sample index')
    # plt.ylabel('distance')
    R = sch.dendrogram(
        Z,
        leaf_rotation=90., # rotates the x axis labels
        leaf_font_size=16., # font size for the x axis labels
        labels=[str(word*100.0) + '%' for word in labels], # label for plot
        color_threshold=None # .5
    )
    ax = plt.gca()
    ax.tick_params(axis='y', labelsize=16)
    plt.show()
    return R


def getxprtnnZeroOne(expReturns=pd.Series):
    ##### JPM Oct 10 2017, Berowne Hlavaty & Robert Smith #####
    # Rescale expected returns from 0 to 1
    if expReturns is None:
        pass
    else:
        idx=expReturns.index
        expReturns = zscore(expReturns)
        expReturns = expReturns/max(abs(expReturns))/2.0+0.5
        expReturns = expReturns / max(abs(expReturns))
        # in case when max(abs()) was -ve, then new max will be < 1.0
        expReturns = pd.Series(expReturns, index=idx)
    return expReturns


def zscore(a, axis=0, ddof=0, keepNaN=False):
    """ NAN Stable Z-Scores"""
    ##### JPM Oct 10 2017, Berowne Hlavaty & Robert Smith #####
    try:
        idx = a.index
    except:
        idx = None
    a = np.asanyarray(a)
    mns = np.nanmean(a, axis=axis)
    sstd = np.nanstd(a=a, axis=axis, ddof=ddof)
    if axis and mns.ndim < a.ndim:
        res = ((a - np.expand_dims(mns, axis=axis)) /
               np.expand_dims(sstd, axis=axis))
    else:
        res = (a - mns) / sstd # RESult
    if not keepNaN:
        res = np.nan_to_num(res) # Default set to zero where was NaN
    res = pd.Series(res, index=idx)
    return res


def correlDist(corr):
    # A distance matrix based on correlation, where 0<=d[i,j]<=1
    # This is a proper distance metric
    return ((1-corr) / 2.) ** .5


def plotCorrMatrix(path, corr, labels=None):
    # Heatmap of the correlation matrix
    if labels is None:
        labels = []
    plt.pcolor(corr)
    plt.colorbar()
    plt.yticks(np.arange(.5, corr.shape[0] + .5), labels)
    plt.xticks(np.arange(.5, corr.shape[0] + .5), labels)
    plt.savefig(path)
    plt.clf()
    plt.close() # reset pylab
    return


def generateData(nobs, size0, size1, sigma1):
    # Time series of correlated variables
    # 1) generating some uncorrelated data
    np.random.seed(seed=12345)
    random.seed(12345)
    x = np.random.normal(0, 1, size=(nobs,size0)) # each row is a variable
    # 2) creating correlation between the variables
    cols = [random.randint(0, size0-1) for i in range(size1)]
    y = x[:,cols] + np.random.normal(0, sigma1, size=(nobs,len(cols)))
    x = np.append(x,y,axis=1)
    x = pd.DataFrame(x, columns=range(1, x.shape[1]+1))
    return x, cols


def getAllocWeights(rtns, exp_rtns=None, riskaversion=0.5):
    ### @tqiu ###
    rtns.fillna(0)
    if exp_rtns is not None:
        exp_rtns.fillna(0)
    cov = rtns.cov()
    corr = rtns.corr()
    dist = correlDist(corr)
    link = sch.linkage(dist, 'average') #'single') #https://towardsdatascience.com/introduction-hierarchical-clustering-d3066c6b560e
    sortIx = getQuasiDiag(link)
    sortIx = corr.index[sortIx].tolist() # recover labels
    ivp = getIVP(cov)
    #cla = getCLA(cov)
    cla = ivp
    hrp = getRecBipart(cov, sortIx)
    harp = getHARP(cov, sortIx, exp_rtns, riskaversion)
    hcp = getClusters(corr)
    return ivp, cla, hrp, harp, hcp


def calcAllocWgts(rtns, lookback=260, numRun=0, exp_rtns=None, riskaversion=0.5, naIgnore=True):
    ### @tqiu ###
    if naIgnore:
        X = rtns
        if exp_rtns is not None:
            EX = exp_rtns
    else:
        X = rtns.dropna(axis=1)
        if exp_rtns is not None:
            EX = exp_rtns[X.columns]
            EX.fillna(method="ffill", inplace=True)
            EX.fillna(method="bfill", inplace=True)
            EX.fillna(value=0, inplace=True)
    if numRun <= 0:
        T = len(X.index)
    else:
        T = lookback + numRun + 1
        X = X[-T:]
        if exp_rtns is not None:
            EX = EX[-T:]
    dfIVP = np.zeros((T,len(X.columns)))
    dfCLA = np.zeros((T,len(X.columns)))
    dfHRP = np.zeros((T,len(X.columns)))
    dfHARP = np.zeros((T,len(X.columns)))
    dfHCP = np.zeros((T,len(X.columns)))
    for t in range(T):
        if t > lookback:
            if exp_rtns is not None:
                ivp_wgt, cla_wgt, hrp_wgt, harp_wgt, hcp_wgt = getAllocWeights(X[t-lookback+1:t], EX[t-1:t], riskaversion)
            else:
                ivp_wgt, cla_wgt, hrp_wgt, harp_wgt, hcp_wgt = getAllocWeights(X[t-lookback+1:t])
            dfIVP[t] = ivp_wgt
            dfCLA[t] = cla_wgt
            dfHRP[t] = hrp_wgt
            dfHARP[t] = harp_wgt
            dfHCP[t] = hcp_wgt
    dfIVP = pd.DataFrame(dfIVP,index=X.index,columns=X.columns)
    dfCLA = pd.DataFrame(dfCLA,index=X.index,columns=X.columns)
    dfHRP = pd.DataFrame(dfHRP,index=X.index,columns=X.columns)
    dfHARP = pd.DataFrame(dfHARP,index=X.index,columns=X.columns)
    dfHCP = pd.DataFrame(dfHCP,index=X.index,columns=X.columns)
    if not naIgnore:
        dfIVP = dfIVP.reindex(columns=rtns.columns)
        dfCLA = dfIVP.reindex(columns=rtns.columns)
        dfHRP = dfIVP.reindex(columns=rtns.columns)
        dfHARP = dfIVP.reindex(columns=rtns.columns)
        dfHCP = dfIVP.reindex(columns=rtns.columns)
    return dfIVP, dfCLA, dfHRP, dfHARP, dfHCP


def main():
    # 1) Generate correlated data
    nobs, size0, size1, sigma1 = 10000, 5, 5, .25
    x, cols = generateData(nobs, size0, size1, sigma1)
    print([(j+1, size0+i) for i,j in enumerate(cols,1)])
    cov = x.cov()
    corr = x.corr()
    # 2) Compute and plot correl matrix
    plotCorrMatrix('HRP3_corr0.png', corr, labels=corr.columns)
    # 3) Cluster
    dist = correlDist(corr)
    link = sch.linkage(dist, 'single')
    sortIx = getQuasiDiag(link)
    sortIx = corr.index[sortIx].tolist() # recover labels
    df0 = corr.loc[sortIx,sortIx] # reorder
    plotCorrMatrix('HRP3_corr1.png', df0, labels=df0.columns)
    # 4) Capital allocation
    hrp = getRecBipart(cov, sortIx)
    print(hrp)
    return


if __name__ == '__main__':
    main()


__all__ = ["getAllocWeights", "calcAllocWgts", 
           "getIVP", "getCLA", "getHRP", 
           "getRecBipart", "getHARP", "getClusters"]
