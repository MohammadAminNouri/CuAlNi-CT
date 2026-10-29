
# Morges, 2016

# C. Cayron 
#

# ==============================================================================
# ==============================================================================
#
# Calculate the orientation relationships between the variants
# In case of phase transition or multiple twinning
# Use the groupoid theory for calculations
# Variants = cosets, Operators = Double-Cosets
# According to the theory devolopped by Cayron, Acta Cryst. A. 2006 and Cayron, Acta Cryst. A. 2007

# ==============================================================================
# ==============================================================================

import sys, os , string
import time
from tkFileDialog import *

v = sys.version[:3]
versionP = "Python" + str(reduce(lambda x,y:x+y, v.split(".")))

try:
    Num = "C:\\"+versionP+"\\lib\\site-packages\\numpy"
    os.chdir(Num)
    disque = "C:"
except:
    try:
        Num = "D:\\"+versionP+"\\lib\\site-packages\\numpy"
        os.chdir(Num)
        disque = "D:"
    except:
        print "je ne trouve pas ou vous avez installe Python!!!!!!!"


print "disque=",disque
print "version=",versionP
PIL= disque+"\\"+versionP+"\\lib\\site-packages\\PIL"
NUM = disque+"\\"+versionP+"\\lib\\site-packages\\numpy"

if PIL not in sys.path: sys.path.append(PIL)
if NUM not in sys.path: sys.path.append(NUM)

from numpy      import *
from numpy.linalg import inv,eig,det
from Tkinter        import *
from Dialog     import Dialog
import tkMessageBox
import pickle
import copy

sys.path.append(disque+"\\"+versionP+"\\arpge\\Algebra")
from CC_AlgebraicTools     import IntegriseAxe, FindSubGroups, cross, calculeRot,normalize, calculeRot_axe_brute
sys.path.append(disque+"\\"+versionP+"\\arpge\\Graphics")
from CC_PlotPoleFigure     import PolFigVariants
from CC_Crystals        import importCryst,cosinus,normalizeCryst, ListEquivRot
from CC_Crystals        import crystWindow, ToFourIndexDir,ToThreeIndexDir,ToFourIndexPlane,ToThreeIndexPlane

try:
    sys.path.append(disque+"\\"+versionP+"\\arpge\\Solid3D_Soya")
    from CC_Transition3D    import  showTransition_Tk
    soya = 1
except:
    soya = 0
    print "Soya3D not updated for Python2.7, should be installed"
    


phaseDirectory      = disque+"\\"+versionP+'\\arpge\\Crystallography\\PhasesCRYST'
transitionDirectory = disque+"\\"+versionP+'\\arpge\\Crystallography\\Transitions'
twinDirectory       = disque+"\\"+versionP+'\\arpge\\Crystallography\\Transitions\\Sigma3'
alphabet = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o",
            "p","q","r","s","t","u","v","w","x","y","z"]
alphabet.extend(["a"+l for l in alphabet])
alphabet.extend(["b"+l for l in alphabet[:26]])


# lalphabet est complete par la suite si necessaire


class ClassOR:
        def __init__(self,name="Burgers",fileMat='Ti_beta',filePrec='Ti_alpha',p1Prec=[0,0,1],p1Mat=[1,1,0],
                     p2Prec= [],p2Mat=[],d1Prec =[1,0,0],d1Mat=[1,-1,1],d2Prec=[],d2Mat=[]):
            
            self.name                   = name
            self.fileMat        = fileMat
            self.filePrec       = filePrec
            self.p1Prec         = p1Prec
            self.p1Mat          = p1Mat
            self.p2Prec         = p2Prec
            self.p2Mat          = p2Mat
            self.d1Prec         = d1Prec    
            self.d1Mat          = d1Mat
            self.d2Prec         = d2Prec    
            self.d2Mat          = d2Mat
            
        def dual(self):
            name = self.name+"_Dual"
            fileMat = self.filePrec
            filePrec = self.fileMat
            p1Prec = self.p1Mat
            p1Mat = self.p1Prec
            p2Prec = self.p2Mat
            p2Mat = self.p2Prec
            d1Prec =  self.d1Mat
            d1Mat =self.d1Prec
            d2Prec = self.d2Mat
            d2Mat = self.d2Prec
            dualOR = ClassOR(name,fileMat,filePrec,p1Prec,p1Mat,p2Prec,p2Mat,
                             d1Prec,d1Mat,d2Prec,d2Mat)
            return dualOR

OR = ClassOR()

class ClassData:
        def __init__(self,name="Burgers",OR=ClassOR()):
            self.OR = OR
        def dual(self):
            name = self.name+"_Dual"
            dualOR = self.OR.dual()
            dualData = ClassData(name,dualOR)
            return dualData
            
Data = ClassData()


varOR=["Burgers","Fe-KS","Fe-NW","Sigma3"]

for ORi in varOR:
    
        OR = ClassOR()
        if ORi== "Burgers":
            name                        = "Burgers"
            fileMat         = 'Ti_beta'
            filePrec                = 'Ti_alpha'
            p1Prec          = [0,0,1]
            p1Mat           = [1,1,0]
            p2Prec          = []
            p2Mat           = []
            d1Prec          = [1,0,0]
            d1Mat           = [1,-1,1]
            d2Prec          = []
            d2Mat           = []

        elif ORi=="Fe-KS":
            name                        ="FeKS"
            fileMat         = 'Fe_gamma'
            filePrec                = 'Fe_alpha'
            p1Prec          = [1,1,0]
            p1Mat           = [1,1,1]
            p2Prec          = []
            p2Mat           = []
            d1Prec          = [1,-1,1]
            d1Mat           = [1,-1,0]
            d2Prec          = []
            d2Mat           = []

        elif ORi=="Fe-NW":
            name                        = "FeNW"
            fileMat         = 'Fe_gamma'
            filePrec                = 'Fe_alpha'
            p1Prec          = [1,1,0]
            p1Mat           = [1,1,1]
            p2Prec          = []
            p2Mat           = []
            d1Prec          = [0,0,1]
            d1Mat           = [-1,1,0]
            d2Prec          = []
            d2Mat           = []

        elif ORi=="Sigma3":
            name                        = "Sigma3"
            fileMat         = 'Fe_gamma'
            filePrec                = 'Fe_gamma'
            p1Prec          = [1,1,1]
            p1Mat           = [-1,-1,-1]
            p2Prec          = []
            p2Mat           = []
            d1Prec          = [1,-1,0]
            d1Mat           = [-1,1,0]
            d2Prec          = []
            d2Mat           = []
            
        OR.name = name
        OR.fileMat = fileMat
        OR.filePrec = filePrec
        OR.p1Prec = p1Prec
        OR.p1Mat = p1Mat
        OR.p2Prec= p2Prec
        OR.p2Mat= p2Mat
        OR.d1Prec = d1Prec
        OR.d1Mat = d1Mat
        OR.d2Prec = d2Prec
        OR.d2Mat = d2Mat

        if ORi=="Burgers": Data.Ti = OR
        elif ORi=="Fe-NW": Data.FeNW = OR
        elif ORi=="Fe-KS": Data.FeKS = OR
        elif ORi=="Sigma3": Data.Sigma3 = OR

choiceOR = "Burgers"

AxeRef1 = [1,0,0]
AxeRef2 = [0,1,0]
AxeRef3 = [0,0,1]

precAxisTolerance   = 0.0001
ErrorNorm       = 0.0001
Identity        = array([[1,0,0],[0,1,0],[0,0,1]])
precAng         = 0.01 # radian
precAxe         = 0.01 # radian

imageName       = 'ebsd'
imageSize       = 512
imageBorder     = 20
diameterMat     = 16
diameterPrec    = 8

colorFlag       = 1 # 0: grey,  1: color
gradientFlag    = 1 # 0: false, 1: true
borderFlag      = 1 # 0: none,  1: grey line
indiceFlag      = 0 # 0: no,    1: indices,     2: small indices
doubleDiffFlag  = 0 # 0: no,    1: double diffraction
variantFlag     = 1 # 0: no,    1: in 1 plots,      2: in diff. plots
DataFlag        = 0 # 0: Ti, 1 : Fe-NW, 2: Fe-KS 
Sauv            = 0 # sauvegarde des resultats dans des fichiers CC_
OrderTwin       = -1 # phase transition or 1,2,3.. Twin
UVError         = "Bad choice for U and V"

os.environ ['PS_PREVIEWER'] = ''

# ==============================================================================
#------------------------------------------------------
def reverse(liste):
    # list dinidices de classment de sort, donne la reciproque
    newlist=[]
    for l in range(len(liste)):
        newlist.append(liste.index(l))
    return newlist

#------------------------------------------------------
def indice(l,a):
    #indice de la matrice a dans la list l
    
    res=-1
    i=1
    for ai in a:
            if allclose(l,ai):res=i
            i=i+1
    return res

#------------------------------------------------------


def Intra(q1,q2):
    #regarde si la list de matrices q1 est dans la list q2

    flag=1
    for m1 in q1:
        flag0=0
        for m2 in q2:
            if allclose(m1,m2):flag0=1
        if flag0==0: flag=0
        
    return flag

#------------------------------------------------------ 
def Inter(q1,q2):
    # regarde si l'interection de deux list contient au moins un element
    
    flag=0
    for m1 in q1:
        for m2 in q2:
            if allclose(m1,m2):
                flag=1
                break
    return flag

#------------------------------------------------------
def EgalListQ(lQ1,lQ2):
    #regarde si deux list de list de matrices lQ1 et lQ2 sont egales

    flag=1
    
    if len(lQ1)!=len(lQ2): flag=0
    else:
        for Q1 in lQ1:
            flag0=0
            for Q2 in lQ2:
                if Intra(Q1,Q2):flag0=1
            flag= flag*flag0
        
    return flag

#------------------------------------------------------

def Inside(lQ0,listlistQ):
    # regarde si une liste de matrices est dans une list de list de matrices

    flag=0
    for lQ in listlistQ:
        if EgalListQ(lQ0,lQ): flag=1
        
    return flag

# ==============================================================================

def CrossSet(lm1,lm2):
    # calcule le produit de deux listes de matrices
    lmprod=[]
    for m1 in lm1:
        for m2 in lm2:
            m = dot(m1,m2)
            flag=1
            for mp in lmprod:
                if allclose(m,mp):flag=0
            if flag: lmprod.append(m)
    return lmprod

# ==============================================================================

def calculeTransf (listmat,TS):
    #TS tenseur de structure

    listres=[]
    for mat in listmat:
        print mat
        res0='I'
        res=0
        mat = array(mat)                
        if det(mat) <0: 
            res0 = "-I"
            mat = dot(-Identity,mat)
                    
        res = calculeRot (mat)
        listres.append([res0,res])
        
    return listres

# ============================================================================== 
##def SortFirst(lrot) :
##    #lrot = [[ang,axe,mat],...]
##    # tri selon les angles


   
# ==============================================================================   
# ==============================================================================

def calculateQuotientGroup (Um, Vm, Up, Vp, GMat, SMat, GPrec, SPrec, symMat, symPrec):
    # opt = Dir pour transition direct
    # Dual pour transition inv
    # Both pour les deux
    # Calcule le Smd, groupe quotient Sm/Smd, les operateurs avec leur code en variants op => {Sm Sm/Smd}
    # ne pas oblier qu'en fait v = Sm/Smd V et op = V-1 {Sm Sm/Smd} V
    
    global Quotient

    Umc = dot (SMat, Um)
    Vmc = dot (SMat, Vm)
    Wm  = dot (inv (SMat), cross (Umc, Vmc))
    
    Upc = dot (SPrec, Up)
    Vpc = dot (SPrec, Vp)
    Wp  = dot (inv (SPrec), cross (Upc, Vpc))

    if det(array([Um,Vm,Wm]))<0: Wm=-Wm #prend un repere direct
    if det(array([Up,Vp,Wp]))<0: Wp=-Wp #prend un repere direct
    
    M = transpose (array ([ normalizeCryst (Um, GMat),
                normalizeCryst (Vm, GMat),
                normalizeCryst (Wm, GMat)]))
            
    P = transpose (array ([ normalizeCryst (Up, GPrec),
                normalizeCryst (Vp, GPrec),
                normalizeCryst (Wp, GPrec)]))
    #M = M(mat->ref), P=P(mat->ref), ref repere intermediaire direct
    if det(M)*det(P)<0.0001:

            print GMat
            print M, det(M)
            print "***"
            print GPrec
            print P,det(P)
            
            P= -P
            print 'erreur programmation: orientation du repere intermediaire R non compatible'
            print "je la corrige"

    #------------------------------------------------------
    
    def CalcSmd(symMat,symPrec,T):
        # calcule le groupe de correspondence
        K=[]
        H=[]
        for symmat in symMat:
            for symprec in symPrec:
                if allclose(dot(dot(inv(T),symmat),T),symprec):
                    K.append(symmat)
                    H.append(symprec)
        print "-----------------"
        print "Sdm = psi(Smd)"
        for h in H : print h
        print "-----------------"
        return K
    #------------------------------------------------------
    
    def QuotientGroup(symMat,Smd):
        # calcule le groupe quotient

        QuotientG=[]
        for symmat in symMat:
            newQ = []
            for s in Smd:
                new =dot(symmat,s)
                newQ.append(new)
            flag=1
            for Q in QuotientG:
                if Inter(Q,newQ): flag=0
            if flag: QuotientG.append(newQ)
            
        return QuotientG
    
    #------------------------------------------------------
      
    def prod(Q1,Q2,groupQ):
        # attention produit particulier utilise pour calculer les operateurs
        
        Q=[]

        for m1 in Q1:
            for m2 in Q2:
                m12 = dot(inv(m1),m2)
                flag=0
                for m in Q:
                    if allclose(m12,m): flag=1
                if flag==0: Q.append(m12)

                
        listQ=[]
        for gQ in groupQ:
            if Intra(gQ,Q): listQ.append(indice(gQ,groupQ))
                
        return listQ

     #------------------------------------------------------
    
    def GroupQCarre(groupQ,opt="normal",Qdual=None):
        # option = normal, sym ou antisym
        
        listsubQ=[]
        complQ = copy.deepcopy(groupQ)  
        Q0 = groupQ[0] #=Smd
        groupQcar = []
        i=1
        for Q1 in groupQ:
            j=1
            lgroupQCar=[]
            for Q2 in groupQ:
                if opt=="normal":
                    listQcar=prod(Q1,Q2,groupQ)
                elif opt=="sym":
                    listQcar=prodSym(Q1,Q2,groupQ)   
                lgroupQCar.append(listQcar)
                j= j+1
            groupQcar.append(lgroupQCar) 
            i=i+1
   
        return groupQcar

    #-------------------------------------------------------------
    
    def calculeTout(M,P,symMat,symPrec,SMat):
        
        mMatPrec = dot(M,inv(P))
               
        print "choiceOR",choiceOR
        if choiceOR == "Sigma3":
            mMatPrec = array([[ 1, -2, -2],[-2,  1, -2],[-2, -2,  1]])
            mMatPrec = dot(1./3,mMatPrec)
        
        Smd= CalcSmd(symMat,symPrec,mMatPrec)

        print "----------------------------------------------------"
        print "Intersection Group H"
        print "matrices------------------"
        #for el in Smd : print el
        #print "symmetries----------------"
        #lrot= calculeTransf (Smd,GPrec)
        #for el in lrot: print el
        print "----------------------------------------------------"
                
        Quotient = QuotientGroup(symMat,Smd)
        # quotient = Sm/Smd
        print "***************************************************"
        print "Nb of left cosets = Nb Variants = ",len(Quotient)
        print "**************************************************"
        i=1
        for el in Quotient:
            print "-----------------------------------------------"
            print "Variant number",i
            i = i+1
            lrot= calculeTransf (el,SMat)
            print lrot
        
        #travail du 7/01/2007 *********************************************
        print "matrice T="
        print mMatPrec
        print
        # calcul des configs equivalentes
        print "***************************************************"
        print "List of equivalent configurations "
        print "**************************************************"
        listConfig=[]
        for sym in symMat:
            config=[]
            for i in range(len(Quotient)):
                var = Quotient[i]
                mvar = var[0]
                newmvar = dot(sym,mvar)
                flag=1
                for j in range(len(Quotient)):
                    for mvarj in Quotient[j]:
                        if allclose(newmvar,mvarj):
                            config.append(j+1)
                            flag=0
                            break
                    if not flag: break
            if config not in listConfig: listConfig.append(config)
        print
        for c in listConfig: print c
        print
        print "*******************************************************"
        print

        # a effacer---
        # pour acceler calcul des macles
        ListBase=[]
        Groupoid = []
        # -------
        
        
        #===========================================================
        print "==================================================="
        print " Calculations of the double-cosets H\G/H" 
        Quotient = array(Quotient) 
        listop=[]
        for dn in Quotient:
            opn=[]
            for s in Smd:
                for m in dn:
                    opel = dot(s,m)
                    opn.append(opel)
            listop.append(opn)      
        listnewop=[]
        for op in listop:
            flag=1
            for opref in listnewop:
                if Intra(op,opref): flag=0
            if flag: listnewop.append(op)
        print "nb of double-cosets = nb operators = ", len(listnewop)
        print "==================================================="
        #===========================================================
        print
        print "calcul par la table di-1 dk = dik exprimee dans M"
        Car= GroupQCarre(Quotient,"normal")
        # Car= table di.dk=> dik dans M = {du}
        # attention muliplier a gauche par mPrecMat et a droite par mMatPrec
        # pour avoir l'operateur
        # Car= GroupQCarre(Quotient,"sym") utile si on veut calculer l'operateur entre di et dk
        # et non l'operateur de di a dk
        for el in Car: print el
        listCar=[]
        for c in Car:
            for cc in c:
                listCar.append(cc)                
        Groupoid=[]
        for cc in listCar:
            if cc not in Groupoid:Groupoid.append(cc)
        print "cardinal Smd/Sm-1 Smd/Sm = nb rotational operators ",len(Groupoid)
        print "-----------------------------------------------------------------"
        i=0
        for el in Groupoid:
            print "Operator number",i,"=viv1 with vi= ", el
            i = i+1
        print "-----------------------------------------------------------------"
        

        #===========================================================

        ListBase=[]
        for op1 in Groupoid:
            ll=[]
            for i in range(len(Car)):
                for j in range(len(Car[0])):
                               if op1==Car[i][j]: ll.append([j+1,i+1])
            ListBase.append([op1,ll])
        #======================================================================
        # jusquici les operateurs sont de type viv1
        # on les mets sous forme v1vi
        print
        print "----------------------------------------------------"
        print "list of the base (source and target) of operators"
        print "----------------------------------------------------"
        print
        Groupoid = []
        i=0
        for el in ListBase :
            print "--------------"
            print "operator",i,"= v1vi with vi = ",
            mm=[]
            for st in el[1]:
                if st[0]==1:mm.append(st[1])
            print mm
            Groupoid.append(mm)
            i = i+1
            x = el[1]
            x.sort()
            print x

        
        #===========================================================
        print
        print " Table of composition di.dk=> operator of dik = di-1 dk"
        print 
        TableComp = []
        for c in Car:
            ligne=[]
            for cc in c:
                op = Groupoid.index(cc)
                ligne.append(op)
            TableComp.append(ligne)
            
        print range(1,len(TableComp)+1)
        print "---------------------------------"
        i=1
        for el in TableComp:
            print i,",",el
            i = i+1
            
        print "-----------------------------------------------------------------"
        print
        print "----------------------------------------------------"
        print "Table of composition of operators developed with variants"
        
        print "line and column of the crossing table = op={du}"
        for op1 in Groupoid:
            print op1,
        print
        TableCompOp=[]
        for op1 in Groupoid:
            print
            double=[]
            for i in op1:
                ligne=[]
                print
                for op2 in Groupoid:
                    op12=[]
                    print ":",  
                    for j in op2:
                        ij = TableComp[i-1][j-1]
                        print ij,
                        op12.append(ij)
                    ligne.append(op12)
                double.append(ligne)
            TableCompOp.append(double)
        print
        print
##            for el in TableCompOp :
##                print
##                for el2 in el: print el2

        print
        print "----------------------------------------------------"
        print "Table of composition of operators simplified"
        print "----------------------------------------------------"
        print
        TableCompOp=[]
        for op1 in Groupoid:
            ligne=[]
            for op2 in Groupoid:
                for i in op1:
                    op12=[]
                    for j in op2:
                        ij = TableComp[i-1][j-1]
                        if ij not in op12: 
                            op12.append(ij)
                ligne.append(op12)
            TableCompOp.append(ligne)
            
        for el in TableCompOp: print el


        return mMatPrec,listConfig, Quotient,Groupoid,ListBase,TableCompOp

    #--------------------------------------------------------------------------

    mMatPrec,listConfigDir,QDir,GrDir,ListDir,TableCompOpDir = calculeTout(M,P,symMat,symPrec,SMat)
    mPrecMat,listConfigDual,QDual,GrDual,ListDual,TableCompOpDual = calculeTout(P,M,symPrec,symMat,SPrec)
    res1 = [mMatPrec,listConfigDir,QDir,GrDir,ListDir]
    res2 = [mPrecMat,listConfigDual, QDual,GrDual,ListDual]
    res= [res1,res2]
    if Sauv: saveDir = transitionDirectory+"\\"+choiceOR
    else: saveDir = transitionDirectory+"\\Current"
    os.chdir (saveDir)
    f = open('CC_PhaseTransition.txt','w')
    pickle.dump([mMatPrec,QDir,QDual,ListDir,GrDir,TableCompOpDir],f)
    f.close()
    f = open('CC_PhaseTransition_inverse.txt','w')
    pickle.dump([mPrecMat,QDual,QDir,ListDual,GrDual,TableCompOpDual],f)
    f.close()

    return res

#==============================================================================      
def InListRot(rot,listrot,symPrec):
    res=0
    ang = rot[0]
    axe=rot[1]
    cond1=0
    for rotref in listrot:
        angref = rotref[0]
        axeref = rotref[1]
        if abs(abs(ang)-abs(angref))<0.0001 : cond1=1
        cond2=0
        for sym in symPrec:
            newaxe = dot(sym,axe)
            p = cross(newaxe,axe)
            if abs(p[0])+abs(p[1])+abs(p[2])<0.0001 :cond2=1
        res = cond1 and cond2
        if res==1: break
    return res
    
#==============================================================================

def InListMatInt(m,listmref,symPrec):
    # regarde si une matrice a coef entier est dans une liste [m,ch]
    # aux symetries pres
    res=-1
    for sym in symPrec:
        newm = dot(sym,m)
        i= 0
        for mref,ch in listmref:
            if allclose(newm,mref):
                res=1
            i = i+1
    return res

#==============================================================================

def SimpleListRot(lrot,symPrec):
    # ne marche que pour les cubiques
    lrotsimpl=[lrot[0]]
    lind=[]
    i=0
    if len(lrot)>1:
        for couple in lrot[1:]:     
            ang = couple[0]
            axe = couple[1]
            for coupleref in lrotsimpl:
                angref=coupleref[0]
                axeref=coupleref[1]
                cond1 = (abs(abs(ang)-abs(angref))<0.0001)
                cond2 = EqualAxe(axe,axeref,symPrec)==1
                cond = cond1 and cond2
                if not cond: lrotsimpl.append([ang,axe])
                else:
                    lind.append(i)
                    break
            i = i+1
            
    return lrotsimpl,lind

#==============================================================================

##def SimpleRotMin(lrot,GPrec):
##    u = array([0,0,1])
##    v = array([1,1,1])
##    lrotsimpl=[]
##    lrotang=[]
##    angmin=180
##    axemin=[0,0,1]
##    for couple in lrot:
##        ang = couple[0]
##        if ang<angmin-0.00001:
##            angmin=ang
##            axemin=couple[1]
##        elif abs(angmin-ang)<0.00001:
##            axe = couple[1]
##            p1 = abs(angle(axe,u,GPrec))
##            p2 = abs(angle(axemin,u,GPrec))
##            if p1 < p2-0.00001:
##                angmin = ang
##                axemin = axe
##            elif abs(p2-p1)<0.00001:
##                p1p = abs(angle(axe,v,GPrec))
##                p2p = abs(angle(axemin,v,GPrec))
##                if p1p < p2p:
##                    angmin = ang
##                    axemin = axe
##
##    return [angmin,axemin]        

#==============================================================================
   
def CalculeListAngles(listDir,QDir,mMatPrec,symPrec,SPrec):
    mPrecMat=inv(mMatPrec)
    listopDir=[]
    lopRot=[]
    print "check and delete"
    print len(symPrec)
    print "==="
    print "SPrec"
    print SPrec
    print "=="
    
    for t in listDir:
        index = t[0]
        op=[]
        i = t[0][0]
        mat = QDir[i-1][0]
        mat = dot(mPrecMat,dot(mat,mMatPrec))
        listopDir.append(mat)
    nop=0
    for mat in listopDir:
        flag0 = 0
        print "------------",nop,"---------------------"
        print mat
        print
        lrot = ListEquivRot(mat,symPrec,SPrec,0) # 1 to save mk 2016
        try:
            if lrot[0][1]: print "rotation ", lrot[0][0], "Deg. /", IntegriseAxe(array(lrot[0][1]))
            else: print "rotation ", lrot[0][0]
        except:
            print "le probleme est ici -----------------"
            print lrot
            
        print

        lrotbis = []
        for rot in lrot:
            lmatrot=[]
            MK = rot[2]
            for syml in symPrec:
                flag2=1
                mkl = dot (inv(syml),dot(MK,syml))
                for mm in lmatrot:
                    if allclose(mm,mkl): flag2=0
                if flag2: lmatrot.append(mkl)
            lrotbis.append([rot[0],rot[1],lmatrot])
        lopRot.append(lrotbis)
        
        nop= nop+1


    return lopRot
                
#==============================================================================

def RotSymMin(op,symPrec):
    # pour un operateur op = une matrice reprensetative de op, calcule toutes les rot
    # et renvoie la min
    lrot=[]
    mat=op
    for sym1 in symPrec:
        mk = dot(sym1,mat)
        rot=calculeRot(mk) 
        if rot:
            ang,axe = rot
            lrot.append([ang,axe,mk])
    lrot.sort()
    lrotbis = []
    for rot in [lrot[0]]: # ne garde que la rotation min
        lmatrot=[]
        MK = rot[2]
        for syml in symPrec:
            flag2=1
            mkl = dot (inv(syml),dot(MK,syml))
            for mm in lmatrot:
                if allclose( mm,mkl): flag2=0
            if flag2: lmatrot.append(mkl)  
        lrotbis.append([rot[0],rot[1],lmatrot]) # ne comporte que la rotation min
    return lrotbis[0]
        
 #==============================================================================
            
def CalculeListTwin(listDir,QDir,mMatPrec,symPrecFloat,GPrec,order):

    # calcule la liste de matrice de passage entre twin0 et twin n
    # ce presente sous une forme de liste de liste
    # donne aussi cette liste sous forme de rotation angle axe
          
    def integrateMat(sigma,mat):
        opred= dot(sigma,mat)
        opred=opred.tolist()
        opred2=[map(round,ligne) for ligne in opred]
        opred3=[map(int,ligne) for ligne in opred2]
        opred3 = array(opred3)
        return opred3

    symPrec=[]
    for sym in symPrecFloat:
        symInt=integrateMat(1,sym)
        symPrec.append(symInt)
    print
    GroupRotPrec = []
    for s in symPrec:
        if det(s)>0: GroupRotPrec.append(s)
    print "group order = ",len(GroupRotPrec)
    for el in GroupRotPrec: print el
    print # je suis ici
    
    #-------------------------------------------------------------------------
    # le programme ne depend pas du choix des representants dans les cosets
    #-----------------------------------------------------------------------
    GroupoidTwin=[]
    lmTwin =[]
    sigma=3
    print "=================================================================="
    for lq in QDir:
        print "*-*-*-*-*-*-*-*-*-*-*-*-*-*-*--*-*-*--*-*--*-*-*-*-*-*-*-*--*-*"
        grtwin=[]
        for q in lq:
            m = dot(q,mMatPrec)
            print
            print integrateMat(sigma,m)
            #print calculeRot(m,GPrec)
            grtwin.append(m)
        q=lq[0]
        mTwin = dot(q,mMatPrec)
        mTwinInt=integrateMat(sigma,mTwin)
        lmTwin.append(mTwinInt)
        GroupoidTwin.append(grtwin)
    print

#---------------------------------------------------------------------------------------------------
    # JE PEUX DONC RENTRER EN FORCE des matrices prises dans chacun des coset
    # ATTENTION LE CHOIX DE LA COMBINAISON DES LISTE AVEC UN
    # DEPEND DU REPRESENTANT DE COSET
    # EX: ce pg donne des resultats faux si on n applique pas les plans mirroirs m1,m2,m3

    print "voici les matrices representatives choisies:"
    print
    t0 = array([[ 1, -2, -2],[-2,  1, -2],[-2, -2,  1]])    # mirroir 1 1 1
    print t0
    print
    t1 = array([[1,-2,-2],[2,2,-1],[2,-1,2]])               # rotation 70.5   0 -1 1
    m1 = array([[1,0,0],[0,0,1],[0,1,0]])                   # + mirroir
    print "rotation 70.5   0 -1 1"
    print t1                                                      
    t1 = dot(t1,m1)
    print "avec mirroir"
    print t1
    print
    t2 = array([[2,2,-1],[-2,1,-2],[-1,2,2]])               # rotation 70.5   -1 0 1
    m2 = array([[0,0,1],[0,1,0],[1,0,0]])                   # + mirroir
    print "rotation 70.5   -1 0 1"
    print t2
    t2 = dot(t2,m2)
    print "avec mirroir"
    print t2
    print
    t3 = array([[2,-1,2],[-1,2,2],[-2,-2,1]])               # rotation 70.5   1 -1 0
    m3 = array([[0,1,0],[1,0,0],[0,0,1]])                   #+ mirroir
    print "rotation 70.5   1 -1 0"
    print t3
    t3 = dot(t3,m3)
    print "avec mirroir"
    print t3
    print
    print det(t0),det(t1),det(t2),det(t3)
    lmTwin=[dot(-1,t0),dot(-1,t1),dot(-1,t2),dot(-1,t3)]
    
#---------------------------------------------------------------------------------------------------
    
    print "-----------------------------------------------------------"
    print "les 4 matrices representant les 4 variants"
    for el in lmTwin:
        print el
        mm = dot(1./3,el)
        r = calculeRot(mm)
        if r:
            print "-I * " , calculeRot(dot(-1.,mm))
        print
    print "la premiere donne la branche de croissance"
    print "les 3 autres fixent la croissance"
    print "il a un plan mirroir qui reste concerve pdt toute une croissance"
    print " il est perpendiculaire a la direction fixe [1,-1,0]"
    print "le nb operateurs au rang n est donc donne par : op(n)=3*op(n-1)-1"
    print "-----------------------------------------------------------"
    print      
    print "-------------------------------"
    print "n=",1,"sigma",3
    print lmTwin[0]
    r3= RotSymMin(dot(1./3,lmTwin[1]),GroupRotPrec)
    print r3
    print
    print "-------------------------------"
    print "n=",2,"sigma",9
    s9= dot(lmTwin[1],lmTwin[1])
    print s9
    r9= RotSymMin(dot(1./9,s9),GroupRotPrec)
    print r9
    ListListOp=[
        [[Identity,[[]]]],
        [[lmTwin[1],[[1],[2],[3]]]],
        [[s9,[[1,1],[2,2],[3,3],[1,2],[2,1],[1,3],[3,1],[2,3],[3,2]]]]
                ]
    ListListRot=[[[0,[1,0,0],GroupRotPrec]],[r3],[r9]]
    if order>1: n= order-1
    else : n=1
    
    for i in range (1,n):
        sigma=power(3,i+2)
        print "--------------------------------------------------------"
        print "n=", i+2,"      sigma",sigma
        print "--------------------------------------------------------"
        print
        listNewOp=[]
        listNewRot=[]
        listLastOp = ListListOp[-1]
        mopfirst = listLastOp[0][0]
        
        lch1 = listLastOp[0][1]
        lch11=[]
        for ch1 in lch1:
            ch11 = copy.deepcopy(ch1)
            ch11.append(ch1[-1])
            lch11.append(ch11)
        opnext=dot(mopfirst,lmTwin[1])
        listNewOp.append([opnext,lch11]) # foreward
        
        mnext = dot(1./sigma,opnext)
        rotmin=RotSymMin(mnext,GroupRotPrec)
        listNewRot.append(rotmin)
        lch12=[]
        lch13=[]
        for ch1 in lch1:
            lind = [1,2,3]
            lind.remove(ch1[-1])
            ch12 = copy.deepcopy(ch1)
            ch12.append(lind[0])# left  
            lch12.append(ch12)         
            ch13 = copy.deepcopy(ch1)
            ch13.append(lind[1])# right = right QUE sur cette branche
            lch13.append(ch13)
        opnext=dot(mopfirst,lmTwin[2])
        lch12.extend(lch13)
        listNewOp.append([opnext,lch12])
        mnext = dot(1./sigma,opnext)
        rotmin=RotSymMin(mnext,GroupRotPrec)
        listNewRot.append(rotmin)
        
        for opInt,lch in listLastOp[1:]:

            #foreward
            newlch=[]
            for ch in lch:
                ind = ch[-1]
                chj = copy.copy(ch)
                chj.append(ind)
                newlch.append(chj) 
            ind = lch[0][-1]
            opnext=dot(opInt,lmTwin[ind])
            mnext = dot(1./sigma,opnext)
            rotmin=RotSymMin(mnext,GroupRotPrec)
            listNewOp.append([opnext,newlch])
            listNewRot.append(rotmin)
            
            # config z comme zigzag
            newlch=[]
            for ch in lch:
                ind0 = ch[-1]
                ind = ch[-2]
                rk=2
                while ind ==ind0:
                    rk=rk+1
                    ind =ch[-rk]
                chj = copy.copy(ch)
                chj.append(ind)
                newlch.append(chj)
            ind0 = lch[0][-1]
            ind = lch[0][-2]
            rk=2
            while ind ==ind0:
                rk=rk+1
                ind =lch[0][-rk]
            opnext=dot(opInt,lmTwin[ind])
            mnext = dot(1./sigma,opnext)
            rotmin=RotSymMin(mnext,GroupRotPrec)
            listNewOp.append([opnext,newlch])
            listNewRot.append(rotmin)
            
            # config c comme circle
            newlch=[]
            for ch in lch:
                ind0 = ch[-1]
                ind = ch[-2]
                rk=2
                while ind ==ind0:
                    rk=rk+1
                    ind =ch[-rk]
                lind=[1,2,3]
                lind.remove(ind0)
                lind.remove(ind)
                ind=lind[0]
                chj = copy.copy(ch)
                chj.append(ind)
                newlch.append(chj)
            ind0 = lch[0][-1]
            ind = lch[0][-2]
            rk=2
            while ind ==ind0:
                rk=rk+1
                ind =lch[0][-rk]
            lind=[1,2,3]
            lind.remove(ind0)
            lind.remove(ind)
            ind=lind[0]
            opnext=dot(opInt,lmTwin[ind])
            mnext = dot(1./sigma,opnext)
            rotmin=RotSymMin(mnext,GroupRotPrec)
            listNewOp.append([opnext,newlch])
            listNewRot.append(rotmin)

        #operateur en double du au plan mirroir, rotation autour [1,-1,0]
        print
        print "-----------nb of operators = ", len(listNewOp)
        print
        for ii in range(len(listNewOp)):
            print listNewOp[ii]
            print listNewRot[ii]
            print
        ListListOp.append(listNewOp)
        ListListRot.append(listNewRot)


    return ListListOp,ListListRot
                
#==============================================================================


def doCalculation (OrderTwin):
    global listVarAmbigue
    # dualflag= 0 direct, 1 inv, 2 both


    Um, Vm, Hm, Km      = array (d1Mat), array (d2Mat), array (p1Mat), array (p2Mat)
    Up, Vp, Hp, Kp      = array (d1Prec), array (d2Prec), array (p1Prec), array (p2Prec)

    print "*",fileMat ,filePrec
    crystM = importCryst(fileMat,phaseDirectory)
    crystP = importCryst(filePrec,phaseDirectory)
    print crystM.el, crystP.el
        
    GMat,  SMat, symMat = crystM.metricTensDir,crystM.structTensDir,crystM.sym()
    GPrec, SPrec, symPrec   = crystP.metricTensDir,crystP.structTensDir,crystP.sym()
    
    print ' metric and structure tensors of parent phase'
    print GMat,'\n',  SMat,'\n' 
    print 'metric and structure tensor of daughter phase'
    print GPrec, '\n',SPrec,'\n'
    
    if (d1Mat and d1Prec) and not (d2Mat and d2Prec):   # d1 &amp; p1
        Vm = dot (inv (GMat), Hm)
        Vp = dot (inv (GPrec), Hp)

    if not (d1Mat and d1Prec):              # p1 &amp; p2
        Um = dot (inv (GMat), Hm)
        Up = dot (inv (GPrec), Hp)
        Vm = dot (inv (GMat), Km)
        Vp = dot (inv (GPrec), Kp)

    if abs (cosinus (Um, Vm, GMat) - cosinus (Up, Vp, GPrec)) > 1e-2:
        print " error in doCalculation"

        
    res = calculateQuotientGroup (Um, Vm, Up, Vp, GMat, SMat, GPrec, SPrec, symMat, symPrec)
        # ensemble QDir = quotient, DrDir = groupoid = ensemble des operateurs, list des bases

    if choiceOR=="Sigma3":
            mMatPrec,listConfigDir, QDir,GrDir,ListDir = res[0]
            lopTwin,lopTwinRot=CalculeListTwin(ListDir,QDir,mMatPrec,symPrec,GPrec,OrderTwin)
            CompositionTwin(lopTwin,lopTwinRot,GPrec, SPrec, symPrec)
            if Sauv:
                os.chdir(twinDirectory)
                print " this table has been saved in a file named CC_listTwin.txt" 
                f = open('CC_listTwin.txt','w')
                pickle.dump(lopTwin,f)
                f.close()

            else:
                os.chdir(transitionDirectory+"\\Current")
                print " this table has been saved in a file named CC_listRot.txt" 
                f = open('CC_listRot.txt','w')
                pickle.dump(lopRot,f)
                f.close()
            print "====================fin=================="

            
    elif choiceOR!="Sigma3":
            mMatPrec,listConfigDir,QDir,GrDir,ListDir = res[0]
            mPrecMat,listConfigDual,QDual,GrDual,ListDual = res[1]
            print "---------------------------------------------------------"
            print "operateurs sous forme d ensemble de matrices de rotation"
            print "---------------------------------------------------------"
            print
            print "--------------DIR----------------------------------------"
            lopRotDir = CalculeListAngles(ListDir,QDir,mMatPrec,symPrec,SPrec)
            print
            print "--------------DUAL----------------------------------------"
            lopRotDual=CalculeListAngles(ListDual,QDual,mPrecMat,symMat,SMat)
            print
            if Sauv:
                os.chdir(transitionDirectory+"\\"+choiceOR)
                print " file created =  'CC_listRotDir.txt'" 
                f = open('CC_listRotDir.txt','w')
                pickle.dump(lopRotDir,f)
                f.close()
                print " file created =  'CC_listRotDual'.txt" 
                f = open('CC_listRotDual.txt','w')
                pickle.dump(lopRotDual,f)
                f.close()
            else:
                os.chdir(transitionDirectory+"\\Current")
                print " file created =  'CC_listRotDir.txt'" 
                f = open('CC_listRotDir.txt','w')
                pickle.dump(lopRotDir,f)
                f.close()
                print " file created =  'CC_listRotDual'.txt" 
                f = open('CC_listRotDual.txt','w')
                pickle.dump(lopRotDual,f)
                f.close()
            print
            print "======================================================================================================"
            print "======================================================================================================"
            print
##            print"---------------------------------------------------------------"
##            print "intersection of operator orbit by methode 1"
##            print"----------------------------------------------------------------"
##            Dir1 = QDir[0] #=Smd
##            Dual1 = QDual[0]
##            i=0
##            orb=[]
##            for qdiri in QDir:
##                i=i+1
##                print "---------------between v1 et v",i,"------------------------------"
##                sol=[]
##                for d1 in Dir1:
##                    d1 = dot(d1,mMatPrec)
##                    for mdiri in qdiri:
##                        di = dot(mdiri,mMatPrec)
##                        j=0 
##                        for qdualj in QDual:
##                            j=j+1
##                            mj=dot(qdualj[0],mPrecMat)
##                            l=0
##                            for qduall in QDual:
##                                l=l+1
##                                ml=dot(qduall[0],mPrecMat)
##                                t=dot(dot(inv(ml),inv(d1)),dot(di,mj))
##                                flag=0
##                                for sm in symMat:
##                                    if allclose(t,sm):flag=1
##                                if flag:
##                                    if [l,j] not in sol: sol.append([l,j])
##                orbit=[]
##                if sol:
##                    for soli in sol:
##                        print "parent solution = ",soli[0],"/v1 et ",soli[1],"/v",i
##                        orbit.append(soli[0])
##                else: print "vide"
##                orb.append(orbit)
##
##            print
##            print"---------------------------------------------------------------"
##            print "intersection of operator orbit by methode 2"
##            print"----------------------------------------------------------------"            
##
##            listopDir=[]
##            for t in ListDir:
##                index = t[0]
##                op=[]
##                for i in t[0]:
##                    for mat in QDir[i-1]:
##                        op.append(mat)
##                listopDir.append(op)
##            n=0
##            listopDual=[]
##            listopDualInv=[]
##            for t in ListDual:
##                #print "operator",n
##                n=n+1
##                index = t[0]
##                #print "entre v1 et ",index
##                op=[]
##                opinv=[]
##                for i in t[0]:
##                    for mat in QDual[i-1]:
##                        mat = dot(mMatPrec,dot(mat,mPrecMat))
##                        op.append(mat)
##                        opinv.append(inv(mat))
##                listopDual.append(op)
##                listopDualInv.append(opinv)
##                    
##            n=0
##            for opdir in listopDir:
##                print "------operator Dir------",n,"---------------"
##                n=n+1
##                m1=0
##                for opdualinv in listopDualInv:
##                    m2=0
##                    for opdual in listopDual:
##                        lmat = CrossSet(opdualinv,opdir)
##                        lmat = CrossSet(lmat,opdual)
##                        if Intra(lmat,symMat): print "= operators Dual m1l and m1j= ", m1,m2
##                        m2=m2+1
##                    m1=m1+1
##            print "-----------------------------------------------------"
##
##            def intersect(l1,l2):
##                res=[]
##                for el1 in l1:
##                    if el1 in l2: res.append(el1)
##                return res
##
##            listVarAmbigue=[]
##
##            def iteration(lista,listtot,orb,listinter):
##                global listVarAmbigue
##                #listtot = listtotale 1,2,...12
##                #lista = list a tester
##
##                for el in listtot:
##                    listinter2 = copy.deepcopy(listinter)
##                    list2= copy.deepcopy(lista)
##                    if el not in list2:
##                        list2.append(el)
##                        listinter2.append(orb[el-1])    
##                        inter = reduce(intersect,listinter2)
##                        #print list2,listinter2, inter
##                        if len(inter)>1:
##                            iteration(list2,range(1,len(orb)),orb,listinter2)
##                            list2.sort()
##                            if list2 not in listVarAmbigue: listVarAmbigue.append(list2)
##                    
##
##            iteration([1],range(1,len(orb)+1),orb,[range(1,len(orb)+1)])
##            print "list of variant groups that can have more than one possible parent"
##            for el in listVarAmbigue: print el
         
            print "==================== END =================="

#===========================================================================================================================        
def CompositionTwin(lopTwin,lopTwinRot,GPrec, SPrec, symPrec):
        global OrderTwin

        print "Calculation of the twin composition table"
        print "OrderTwin = ",OrderTwin
        
        if OrderTwin==1:OrderTwin=6 #mini pour que ca marche

        #=================================================================
        # ------ classe dabord sur les angles-------------------------

        lopTwinClass0 = []
        for i in range(len(lopTwin)):
            #print "----------order ",i," ---------------"
            lind = []
            lop = lopTwin[i]
            loprot = lopTwinRot[i]
            lang=[]
            for j in range(len(lop)): #1er passage pour avoir la liste des angles
                ang,axe,lmat = loprot[j]
                lang.append(ang)
            lang.sort()
            lopclass=[0 for k in range(len(lop))]
            loprotclass=[0 for k in range(len(lop))]
            for j in range(len(lop)):
                op,lch = lop[j]
                ang,axe,lmat = loprot[j]
                indice = lang.index(ang)
                lopclass.pop(indice)
                lopclass.insert(indice,[op,lch,ang,axe])
                lang.pop(indice)
                lang.insert(indice,-1) # pour eviter de reprendre le meme angle
                lind.append(indice)
            lopTwinClass0.append(lopclass)
            
        #=================================================================


         # -----------------------------------------------------------
##                print "     after classsing the angles        "
##                i = -1
##                for lop in lopTwinClass0:
##                    i = i+1
##                    print "----------- ORDER  ", i,"   ---------------"
##                    print "nb=",len(lop)
##                    for op,lch,ang,axe in lop:
##                        print op
##                        print lch
##                        print ang,axe
##                        print
##                print "------"
        # -----------------------------------------------------------
        # permet de couper la liste a OrderTwin
        lopTwinClassBis = []
        i = -1
        for lop in lopTwinClass0:
            i = i+1
            if i> OrderTwin:break
            lopbis = copy.deepcopy(lop)
            lopTwinClassBis.append(lopbis)
        lopTwinClass0 = lopTwinClassBis

        #==========================================================================                    
        # ----------------classe ensuite sur les axes  -------------------------
        # fait un double classment
        # selon p1 = produit scalaire avec [111]
        # puis selon p2 produit scalaire avec [001]
        # 3 jours de boulots il faudrait simplifer les notations
        # car ca sera difficilement relisible
        
        i = -1
        U0 = array([1,1,1])# sert de reference poir classer les axes
        U0bis = array([0,0,1])# sert de seconde reference poir classer les axes
        lopTwinClass = []
        for lop in lopTwinClass0:
            i = i +1
            lastaxe = array([1,1,1]) 
            lastang = 9999
            nbpas = len(lop)
            listp1 = []
            listp1bis = []
            #print "lp1",listp1
            for j in range(nbpas):
                op,lch,ang,axe = lop[j]
                if abs(ang-lastang)<0.00001:
                    p1max = -9999
                    p1bismax = -9999
                    if listp1[-1]== -1:
                        preaxe = lop[j-1][3] #calcul pour le precedent 
                        for sym in symPrec:    
                            saxe = dot(sym,preaxe)
                            p1 = abs(dot(normalize(saxe),U0)) # mimique un angle avec U0
                            p1bis = abs(dot(normalize(saxe),U0bis)) # mimique un angle avec U1
                            if p1 > p1max: p1max =p1
                            if p1bis > p1bismax: p1bismax =p1bis
                        listp1.append(p1max)
                        listp1bis.append(p1bismax)
                    p1max = -9999
                    p1bismax = -9999
                    for sym in symPrec:    
                        saxe = dot(sym,axe)
                        p1 = abs(dot(normalize(saxe),U0))
                        p1bis = abs(dot(normalize(saxe),U0bis))
                        if p1 > p1max: p1max =p1
                        if p1bis > p1bismax: p1bismax =p1bis
                    listp1.append(p1max)
                    listp1bis.append(p1bismax)
                else :
                    listp1.append(-1)# les -1 serviront a separer les angles differents
                    listp1bis.append(-1)
                    
                lastang = ang
                
##                    print "****************************************"
##                    print "listp1",i,listp1
##                    print "listp1bis",i,listp1bis
##                    print"****************************************"
            lopnew= copy.deepcopy(lop)
            lastp1=-1
            nbsep=0
            for k in range(len(listp1)):
                p1 = listp1[k]
                if p1!=-1 and lastp1==-1: # une serie commence il faut la classer en ft des axes cad des p1
                    nbsep = nbsep+1
                    listint = []
                    listintBIS = []
                    l=0
                    try:
                        while listp1[k+l]!=-1:     
                            listint.append(listp1[k+l])
                            listintBIS.append(listp1bis[k+l])
                            l=l+1
                    except: pass  #a verifier
                    listintsort = copy.deepcopy(listint)
                    listintsort.sort()

                    l=0
                    lindp1=[]  # liste des indices de classment de listp1
                    listintsortbis=copy.deepcopy(listintsort) 
                    for p in listint:
                        ind = listintsortbis.index(p)
                        lindp1.append(ind)
                        listintsortbis[ind]=-1 #pour eviter de le reprendre
                        l=l+1  
                    
                    lindp1rev=reverse(lindp1)

                    listintBISsort=[]
                    for ind in lindp1rev:
                        listintBISsort.append(listintBIS[ind])
                    listintBIS = listintBISsort
##


                    #repere tous les p1 de meme valeurs (qui se suivent dans lp1sort
                    # les indices sont mis dans listint2
                    # en parallele fait une liste des p2 dans listint3
                    listint2 = []
                    listint3 = []
                    ip1 = 0
                    while ip1 < len(listintsort):
                        p1int = listintsort[ip1]
                        ip2=1
                        listint2bis=[lindp1rev[ip1]]
                        listint3bis=[listintBIS[ip1]]
                        if ip1+1 < len(listintsort):
                            newp1int = listintsort[ip1+1]
                        else :
                            listint2.append(listint2bis)
                            listint3.append(listint3bis)
                            break
                        while abs(p1int -newp1int)<0.000001:  
                            listint2bis.append(lindp1rev[ip1+ip2])
                            listint3bis.append(listintBIS[ip1+ip2])
                            ip2 = ip2+1
                            if ip1+ip2 < len(listintsort):
                                newp1int = listintsort[ip1+ip2]
                            else :break
                        ip1 = ip1+ip2
                        listint2.append(listint2bis)
                        listint3.append(listint3bis)

                    
                    # trie selon p1 et p2 en meme temps
                    l=0
                    lt = 0
                    for lip1 in listint2:
                        listint3sort = copy.deepcopy(listint3[l])
                        listint3sort.sort()

                        lindl = []
                        for p2 in listint3[l]:
                            indl = listint3sort.index(p2)
                            listint3sort[indl]=-1
                            lindl.append(indl)

                        lip2 = [-1 for z in range(len(lip1))]
                        y=0
                        for indl in lindl:
                            lip2[indl]=lip1[y]
                            y=y+1

                        for ip2 in lip2:
                            lopnew[k-nbsep+lt]=lop[k-nbsep+ip2]
                            lt = lt+1
                        l = l + 1                               

                else: pass
                lastp1 = p1
                
            lopTwinClass.append(lopnew)

        lopTwinClass0=[] # pour remettre la memoire a zero
        #========================================================================== 


        #========================================================================== 
        # ------------------NOMME LES OPERATEURS-------------------------------------
        print
        i = -1
        lopTwinNom= []
        for lop in lopTwinClass:
            i = i +1
            lopnom=[]
            j=-1
            for op,lch,ang,axe in lop:
                j = j+1
                flaga=0
                if abs(ang-lastang)<0.00001:
                    for sym in symPrec:
                        saxe = dot(sym,axe)
                        if allclose(saxe,lastaxe):
                            flaga=1
                            break
                    if flaga:
                        j = j-1
                        try:
                            nom = str(i)+"-"+alphabet[j]
                            lopnom[-1][4] = str(i)+"+"+alphabet[j]
                        except : # si alphabet trop court alphab est aggrandi par ab,ac
                            lettre = alphabet[-1][0]
                            poslet = alphabet.index(lettre)
                            try:
                                lettresuiv = alphabet[poslet+1]
                                alphabet.extend([lettresuiv+l for l in alphabet[:26]])
                                nom = str(i)+"-"+alphabet[j]
                                lopnom[-1][4] = str(i)+"+"+alphabet[j]
                            except:
                                nom =""
                                print "alphabet is too short!!"

                    else :
                        try:
                            nom = str(i)+alphabet[j]
                        except :
                            lettre = alphabet[-1][0]
                            poslet = alphabet.index(lettre)
                            try:
                                lettresuiv = alphabet[poslet+1]
                                alphabet.extend([lettresuiv+l for l in alphabet[:26]])
                                nom = str(i)+alphabet[j]
                            except:
                                nom =""
                                print "alphabet is too short!!"
                            
                else :
                    try : nom = str(i)+alphabet[j]
                    except : nom =""
                lopnom.append([op,lch,ang,axe,nom])
                lastaxe = axe
                lastang = ang

            lopTwinNom.append(lopnom)
        print
        
        lopTwinClass=[] # pour remettre la memoire a zero


        #--------------------------------------------------------------------
        # RAJOUTE LES CHEMINES QUI COMMENCENT SUR LAUTRE BRANCHE PAR 0
        #--------------------------------------------------------------------

        lopTwinNom2 = []
        print"++++++++++++++++++ Add branches to the new branch ++++++++++++++++++++++"
        # met 0 au debut
        # et change le sens de rotation, cad 1,2,3 en 1,3,2
        i=0
        for lop in lopTwinNom:
            print
            print "----------- order    ",i," --------------"
            i = i +1
            newlop=[]
            for op,lch,ang,axe,nom in lop: 
                newlch=[]
                for ch in lch:
                    newch = copy.deepcopy(ch)
                    if newch: newch[0]=0
                    if newch not in newlch: newlch.append(newch)
                newlch2= copy.deepcopy(lch)
                newlch2.extend(newlch)
                newlop.append([op,newlch2,ang,axe,nom])
            lopTwinNom2.append(newlop)
        lopTwinNom = lopTwinNom2
            

        #--------------------------------------------------------------------
        print
        print"++++++++++++++++++ OPERATORS RANKED AND NAMED ++++++++++++++++++++++"
        i=0
        for lop in lopTwinNom:
            print
            print "----------- order    ",i," --------------"
            i = i +1
            for op,lch,ang,axe,nom in lop:
                print
                print nom
                print op
                print lch
                print ang,axe
        print
        print " this table has been saved in a file named CC_listTwin_Class.txt" 
        f = open('CC_listTwin_Class.txt','w')
        pickle.dump(lopTwinNom,f)
        f.close()

        #===========================================================================
        # -------- remet les operateurs dans lordre initial mais avec les bons noms
        # --------inverse la liste dindices pour revenir au classement premier

        lopTwinArbre=[]
        i = -1
        for lop in lopTwin:
            i = i +1
            if i > OrderTwin: break
            loparb= []
            for op,lch in lop:
                lopclass = lopTwinNom[i]
                for opc,lchc,angc,axec,nomc in lopclass:
                    if lch[0]==lchc[0]:
                        loparb.append([opc,lchc,angc,axec,nomc])
                        break
            lopTwinArbre.append(loparb)

        
        #--------------------------------------------------------------------
        print
        print"++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
        print "----------------LIST OF OPERATOR TREES ---------------"
        print"++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
        i = -1
        for lop in lopTwinArbre:
            i = i +1
            print "------------------ order ",i," ------------------"
            print
            j=-1
            for op,lch,ang,axe,nom in lop:
                j=j+1
                print nom
                if j==1: print "--------------------------------"
                elif (j-1)%3==0: print "--------------------------------"

                
        print"++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
        print
        print " this table has been saved in a file named CC_listTwin_Arbre.txt" 
        f = open('CC_listTwin_Arbre.txt','w')
        pickle.dump(lopTwinArbre,f)
        f.close()

        print                  
        print"++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
        print"++++++++++++++++++ TABLE DE COMPOSITION ++++++++++++++++++++++"
        print"++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
        print
        #-----------------------------------------------------------


        def chgt(indref1):
        #fait le changt de repere des etapes suivantes
            if indref1==1:listchgt = [1,0,3,2] 
            elif indref1==2:listchgt = [2,3,0,1]
            elif indref1==3:listchgt = [3,2,1,0]
            elif indref1==0: print "ERRRRRREUR dans chgt"
            return listchgt

        def reindex(ch):
            #reindex la fin de liste apres le 0
            newch= copy.deepcopy(ch)
            while 0 in newch[1:]:
                remi = newch[1:].index(0)+1
                if remi != len(newch)-1:
                    indch = newch[remi-1]
                    listchgt = chgt(indch)
                    newch[remi+1]=listchgt[newch[remi+1]]
                newch.pop(remi-1)
                newch.pop(remi-1)
                
            return newch
                 
        def combineChem(lchi,lchj):
            lchij=[]                 
            for chi in lchi:
                if chi[0]!=0:
                    for chj in lchj:
                        #flag=0
                        chij = copy.deepcopy(chi)   
                        chij.extend(chj)
                        if chj[0]==0: #les 0 dans le repere originel signifie retour au passe
##                                    print
##                                    print chi
##                                    print chj
                            chij = reindex(chij)
                        if chij not in lchij:
                            lchij.append(chij)
            
            return lchij
        

        #-----------------------------------------------------------        
        def identifOp(lchij):
            lnom=[]
            for chij in lchij:
                for lopi in lopTwinNom:
                    for opi,lchi,angi,axei,nomi in lopi:
                        for chi in lchi:
                            #pp = [0]
                            #chk = pp.extend(chi[:-1]) # solution autre branche [0,1,1,..]
                            if chij== chi: #or chij== chk:
                                if nomi not in lnom: lnom.append(nomi)
                                break

            return lnom
        #-----------------------------------------------------------
       
        def classNom(lnom):
            # classe une liste de noms de type 4-a etc..

            def lecture(let):
                count = 0
                ord = int(let[0])
                if let[1]=="+" or let[1]=="-": pos=let[2]
                else: pos=let[1]
                pos = alphabet.index(pos) 
                return [ord,pos]

            def reorder(list0):
                
                def doublein(l):
                    res=-1
                    flag=0
                    for i in range(len(l)):
                        if l.count(l[i])>1:
                            res=i
                            flag=1
                            break
                        if flag: break
                    return res
                        
                list=copy.deepcopy(list0)
                i = doublein(list)
                while i!=-1:
                    for j in range(len(list)):
                        if i!=j and list[j]>=list[i]: list[j]=list[j]+1
                    i = doublein(list)
                    
             
                list2 = copy.deepcopy(list)
                list3 = [0 for i in range(len(list))]
                for i in range(len(list)):
                    mini = min(list2)
                    ind = list.index(mini)
                    list3[ind]=i
                    list2[ind]=99999
                    
                return list3
            #------------------------------------------------------

            newlnom=[[] for i in range(len(lnom))]
            lclass= []
            for nom in lnom:
                order,lettre  = lecture(nom)
                lclass.append([order,lettre])
            firstclass=[]
            for ni,nj in lclass:
                firstclass.append(ni)
            firstclass.sort()
            
            #premier classement sur lordre
            for nom in lnom:
                ind = firstclass.index(int(nom[0]))
                firstclass[ind]=-1 # pour eviter de reprendre le meme
                newlnom[ind]=nom
            #second classement sur la lettre
            newlnom2 = copy.deepcopy(newlnom)
            i=0
            lastorder=-1
            while i< len(newlnom):
                order,lettre = lecture(newlnom[i])
                secondclass=[]
                for j in range(len(newlnom)):
                    if i+j>= len(newlnom): break
                    neworder,newlettre = lecture(newlnom[i+j])
                    if order==neworder: secondclass.append(newlettre)
                    else: break       
                secondclass = reorder(secondclass)
                #secondclass.sort()
                k=0
                for ind in secondclass:
                    newlnom2[i+ind] = newlnom[i+k]
                    k=k+1    
                i = i + j         
                
            return newlnom2
                               
        #-----------------------------------------------------------
        
        limite = len(lopTwinNom)/2
        print limite
        i = 0
        TabComp=[]
        for lopi in lopTwinNom[1:]:
            i = i + 1
            if i>limite: break 
            TabComp3=[]
            print "-----------------------------------------------------"
            for opi,lchi,angi,axei,nomi in lopi:
                j=0
                ki=-1
                TabComp2=[]
                for lopj in lopTwinNom[1:]:
                    j = j + 1
                    if j>limite: break
                    print 
                    ki = ki +1
                    kj= -1
                    TabComp1=[]
                    for opj,lchj,angj,axej,nomj in lopj:
                        kj = kj + 1
                        lchij = combineChem(lchi,lchj)
                        nomij = identifOp(lchij)
                        #nomij.append("0z") # sinon je sais pas pourquoi ca marche pas
                        nomij = classNom(nomij)
                        print
                        print nomi," > ",nomj," = ",nomij
                        TabComp1.append(nomij)
                    TabComp2.append(TabComp1)
                TabComp3.append(TabComp2)
            TabComp.append(TabComp3)
                
            print "-----------------------------------------------------"
        print
        print "============================================================"
        print " ---------------TO EXPORT THE DATA INTO A WORD FILE---------------"
        print "  creer un tableau carree de la dimension = sqrt(nbligne)   "
        print "  selectionner le tableau et faire un copier-coller  "
        print "============================================================"
        
        for comp in TabComp:
            for list in comp:
                for lnom in list:
                    for nom in lnom: print nom

            
        print "---------------END------------------------------"

        f = open('CC_listTwinTabCompo.txt','w')
        pickle.dump(TabComp,f)
        f.close()
                    

    
# ==============================================================================

def TkVariants  (frame):

    global choiceOR, OrderTwin, entry_order

    if __name__ == '__main__':
        root2 = Frame(frame)
        root2.grid(row=1,column=0)
    else:
        root2=Toplevel(frame)
             

    def makeWidgets (self):
            global choiceOR, valOR, aframe, OrderTwin, entry_order,OROther, OR 
            valOR=[1,1,1]

            #--------------------------------------------------------------------------------------------------------
            def opencrystal(event):
                crystM = importCryst(fileMat,phaseDirectory)
                root3=Toplevel(root2)
                crystWindow(crystM,root3,0) 

            #--------------------------------------------------------------------------------------------------------
            def OpenTrans():
                global valOR,choiceOR,OROther 
                os.chdir (transitionDirectory+"\\"+"ListAllTransitions")
                oname = askopenfilename(filetypes=[("text files","*.trans"),("others","*.*")])
                if oname :
                    f = open(oname,'r')
                    info = pickle.load(f)
                    name,fileMat,filePrec,p1Prec,p1Mat,p2Prec,p2Mat,d1Prec,d1Mat,d2Prec,d2Mat = info
                    f.close()
                    OROther = ClassOR()
                    choiceOR = name
                    OROther.name = name
                    OROther.fileMat = fileMat
                    OROther.filePrec = filePrec
                    OROther.p1Prec = p1Prec
                    OROther.p1Mat = p1Mat
                    OROther.p2Prec= p2Prec
                    OROther.p2Mat= p2Mat
                    OROther.d1Prec = d1Prec
                    OROther.d1Mat = d1Mat
                    OROther.d2Prec = d2Prec
                    OROther.d2Mat = d2Mat
                    buildaframe(self)
            #--------------------------------------------------------------------------------------------------------
            def buildaframe(self):
                global valOR, OrderTwin, OR
                if choiceOR=="Burgers":
                    OR = Data.Ti
                    valOR=[0,1,2,3]
                elif choiceOR=="Fe-NW":
                    OR = Data.FeNW
                    valOR=[1,0,2,3]
                elif choiceOR=="Fe-KS":
                    OR = Data.FeKS
                    valOR=[1,2,0,3]
                elif choiceOR=="Sigma3":
                    OR = Data.Sigma3
                    valOR=[1,2,3,0]
                else:
                    OR = OROther
                    valOR=[1,1,1,1]
                    
                fileMat= OR.fileMat 
                filePrec = OR.filePrec 
                p1Prec = OR.p1Prec
                p1Mat = OR.p1Mat
                p2Prec = OR.p2Prec
                p2Mat = OR.p2Mat
                d1Prec = OR.d1Prec  
                d1Mat = OR.d1Mat  
                d2Prec = OR.d2Prec 
                d2Mat = OR.d2Mat

                print choiceOR
                print fileMat,filePrec
                aframe = Frame (self, cursor='crosshair', borderwidth=2)
                aframe["relief"]=RAISED
                aframe.grid (row=1, column=0, ipady=5,ipadx = 5,sticky = "EW")
                l = Label(aframe, text = "Phase Transition Information/Creation")
                l["foreground"] = "black"
                l["background"] = "grey"
                l.grid(row=0,column=1,columnspan=2)
                opencrystalframe = Button (aframe, text = "Open/Create a Crystal", borderwidth=2,foreground="blue", background="grey")
                opencrystalframe.bind("<Button>", opencrystal)
                opencrystalframe.grid (row=0, column=3, ipady=5,ipadx = 5,sticky = "EW")
                l = Label(aframe, text = " ").grid(row=1,column=1,columnspan=2)

                variables = [   ['Transition Name',     choiceOR,       0   ],
                                ['Phase Directory',     phaseDirectory,     0   ],
                                ['Parent',          fileMat,        0   ],
                                ['Daughter',            filePrec,       0   ],
                                [],
                                ["Fill either p1_daughter // p1_parent or d1_daughter // d2_parent for two directions :",       -1  ],
                                ['(p1 Daughter plane)',     `p1Prec`[1:-1],     1   ],
                                [' // (p1 Parent plane)     ',  `p1Mat`[1:-1],      2   ],
                                ['(p2 Daughter plane)',     `p2Prec`[1:-1],     1   ],
                                [' // (p2 Parent plane)     ',  `p2Mat`[1:-1],      2   ],
                                ['[d1 Daughter axis] ',     `d1Prec`[1:-1],     1   ],
                                [' // [d1 Parent axis]      ',  `d1Mat`[1:-1],      2   ],
                                ['[d2 Daughter axis] ',     `d2Prec`[1:-1],     1   ],
                                [' // [d2 Parent axis]      ',  `d2Mat`[1:-1],      2   ],
                                [],
                                ['Axis Tolerance',      precAxisTolerance,  0   ],
                                []]
                
                self.fields = []
                k = 2
                for i in range (len (variables)):
                        if not variables[i]:
                                Label (aframe, text="").grid (row=k, column=0, columnspan=4)
                                k=k+1
                        elif variables[i][1] == -1:
                                Label (aframe, text=variables[i][0]).grid (row=k, column=0, columnspan=4)
                                k=k+1       
                        else:
                                if not variables[i][2]:
                                        col1    = 0
                                        col2    = 2
                                        colsp   = 2
                                        width   = 40
                                else:
                                        width   = 19
                                        col1    = 2*variables[i][2]-2
                                        col2    = col1+1
                                        colsp   = 1
                                Label ( aframe,
                                        text=variables[i][0],
                                        relief=RIDGE,
                                        width=width).grid (row=k, column=col1, sticky=W, columnspan=colsp)
                                entry = Entry (aframe, width=width)
                                if variables[i][0]=="Phase Directory":
                                    entry.insert(0, variables[i][1]) #a afficher a partir de la fin du nom du chemin
                                else: entry.insert (0, variables[i][1])
                                entry.grid (row=k, column=col2, columnspan=colsp)
                                self.fields.append (entry)
                                
                                if (variables[i][2] == 0 or variables[i][2] == 2): k=k+1
                return aframe               

            #--------------------------------------------------------------------------------------------------------
            def refreshWidgetTi():
                global choiceOR, aframe
                choiceOR="Burgers"
                aframe.destroy()
                aframe = buildaframe(self)
            def refreshWidgetFeNW():
                global choiceOR, aframe
                choiceOR="Fe-NW"
                aframe.destroy()
                aframe = buildaframe(self)
            def refreshWidgetFeKS():
                global choiceOR, aframe
                choiceOR="Fe-KS"
                aframe.destroy()
                aframe = buildaframe(self)
            def refreshWidgetSigma3():
                global choiceOR, aframe, OrderTwin
                choiceOR="Sigma3"
                attr_order(0)
                aframe.destroy()
                aframe = buildaframe(self)
            def calcSauv():
                global Sauv
                Sauv = (Sauv+1)%2         
            def attr_order(event):
                global OrderTwin
                OrderTwin = string.atoi(entry_order.get())
            #--------------------------------------------------------------------------------------------------------

     
            title = Label ( self,
                            text="GenOVa: Generation of Orientational Variants \n"+
                            "C. Cayron 2016 \n",
                            fg='darkred', cursor='crosshair')
            title.grid (row=0, column=0, columnspan =2, ipady=5)

            aframe = buildaframe(self)
            
            self.tkColorFlag    = IntVar()
            self.tkColorFlag.set(colorFlag)
            
            self.tkGradientFlag = IntVar()
            self.tkGradientFlag.set(gradientFlag)
            
            self.tkBorderFlag   = IntVar()
            self.tkBorderFlag.set(borderFlag)
            
            self.tkIndiceFlag   = IntVar()
            self.tkIndiceFlag.set(indiceFlag)

            self.tkDDFlag   = IntVar()
            self.tkDDFlag.set(doubleDiffFlag)

            self.tkVariantFlag  = IntVar()
            self.tkVariantFlag.set(variantFlag)
            
            self.tkSauv = IntVar()
            self.tkSauv.set(Sauv)

            self.tkDataFlag = IntVar()
            self.tkDataFlag.set (DataFlag)
#               #******************************************************************************************
            bframe = Frame (self, cursor='crosshair',borderwidth = 2)
            bframe["relief"]=RAISED
            bframe.grid (row=2, column=0,columnspan=2, ipady=5,ipadx = 5,sticky = "EW")
            
            l = Label (bframe, text="Load Classical Phase Transition",foreground = "red")
            l["foreground"] = "black"
            l["background"] = "grey"
            l.grid (row=0, column=0,columnspan=2)
            btrans = Frame (bframe)
            btrans.grid (row=1, column=0,ipady=5,columnspan=3, sticky="N")
            Radiobutton (btrans, text="Burgers",variable=self.tkDataFlag, value=valOR[0],command=refreshWidgetTi).grid (row=1, column=0)
            Radiobutton (btrans, text="Fe-NW",variable=self.tkDataFlag, value=valOR[1],command=refreshWidgetFeNW).grid (row=1, column=1)
            Radiobutton (btrans, text="Fe-KS",variable=self.tkDataFlag, value=valOR[2],command=refreshWidgetFeKS).grid (row=1, column=2)
            b = Button (btrans, text="Others", background = "grey", foreground = "black",command = OpenTrans)
            b.grid (row=2, column=0,columnspan=3,sticky = "EW")

            r = Radiobutton (btrans, text="Sigma3 Multiple Twinning",variable=self.tkDataFlag, value=valOR[3],command=refreshWidgetSigma3)
            r.grid (row=1, column=3,columnspan=2,ipadx=20)
            Label (btrans, text="Order").grid (row=2, column=3)
            entry_order = Entry (btrans, text="Ordre",width=3)
            entry_order.insert(0,1)
            entry_order.bind("<Return>",attr_order)
            entry_order.grid (row=2, column=4)      

            #******************************************************************************************
            brun = Frame (self, cursor='crosshair',borderwidth=2)
            brun["relief"]=RAISED
            brun.grid (row=4, column=0,columnspan=2,ipady=5,sticky="EW")
            
            Checkbutton (brun, text="Save Results", variable=self.tkSauv,command = Sauv).grid (row=5, column=0,columnspan=2,sticky="W") 
            Label(brun,text="                 ",width=15).grid(row=3,column=3,sticky="W")
            Label(brun,text="Generate",background = "grey", foreground = "red", width=15).grid(row=2,column=3,sticky="E")
            Button (brun, text='Variants and Operators', width= 20,background = "grey", foreground = "red",command=tkCalculation).grid (row=2, column=4,sticky="W")
            Label(brun,text="Visualize",background = "grey", foreground = "blue", width=15).grid(row=3,column=3,sticky="E")
            Button (brun, text="Calculator",  width= 20, background = "grey", foreground = "blue",command = ShowCalculator).grid (row=3, column=4,sticky="W")
            Button (brun, text="Operator Tables",  width= 20, background = "grey", foreground = "blue",command = ShowTables).grid (row=4, column=4,sticky="W")
            if soya:
                Button (brun, text="Variants in 3D",  width= 20, background = "grey", foreground = "blue",command = Draw3D).grid (row=5, column=4,sticky="W")       
            Button (brun, text="Pole Figure",  width= 20, background = "grey", foreground = "dark green",command = DrawPloleFig).grid (row=6, column=4,sticky="W")
            Label(brun,text="Simulate",background = "grey", foreground = "dark green",width=15).grid(row=6,column=3,sticky="E")
            Button (brun, text="Electron Diffraction",  width= 20, background = "grey", foreground = "dark green",command = DrawSAED).grid (row=7, column=4,sticky="W")
            #Button (brun, text='Quit',  width= 20,command=self.quit).grid (row=5, column=4,sticky="W")

    def infobox (self, title, text, bitmap = '', strings = ('OK',)):
            Dialog (self,
                    title=title, text=text, bitmap=bitmap, default=0, strings=strings)
            
    #----------------------------------------------------------------------------------------------------
            
    def ShowCalculator():

        #sys.path.append(disque+"\\"+versionP+"\\arpge\\Graphics\\Solid3D_Soya")
        from CC_TabToFrame    import  CreateMatrixTransition
        choiceOR= str(root2.fields[0].get ())
        CreateMatrixTransition(choiceOR,root2)
        
    #----------------------------------------------------------------------------------------------------
            
    def ShowTables():

        #sys.path.append(disque+"\\"+versionP+"\\arpge\\Graphics\\Solid3D_Soya")
        from CC_TabToFrame    import  CreateTabTransition
        choiceOR= str(root2.fields[0].get ())
        CreateTabTransition(choiceOR,root2)
            
    #----------------------------------------------------------------------------------------------------
    def DrawSAED():
        
        sys.path.append(disque+"\\"+versionP+"\\arpge\\SimulDiff")
        from CC_Diffraction import TkDiffraction  
        choiceOR= str(root2.fields[0].get ())
        TkDiffraction  (root2,choiceOR)
        #print " SELECTED AREA ELECTRON DIFFFRACTION A FAIRE!!!!"
        
    #----------------------------------------------------------------------------------------------------
    def Draw3D():

        choiceOR= str(root2.fields[0].get ())
                
        if choiceOR=="Sigma3":
            Directory   = disque+"\\"+versionP+'\\arpge\\Crystallography\\Transitions\\Sigma3'
            os.chdir (Directory)
            f = open('CC_listTwin_Arbre.txt','r')
            lopTwinRat=pickle.load(f)
            f.close()
            showTransition_Tk(root2,choiceOR,lopTwinRat)

        else:
            Directory = disque+"\\"+versionP+"\\arpge\\Crystallography\\Transitions\\" + choiceOR
            try: os.chdir (Directory)
            except: os.chdir (disque+"\\"+versionP+"\\arpge\\Crystallography\\Transitions\\Current")

            f = open('CC_PhaseTransition.txt','r')
            mMatPrec, QDir, QDual,TabVar,listOpTarget,Tab = pickle.load(f) 
            f.close()

##            i=0
##            for lmat in QDir:
##                i = i+1
##                for mat in lmat:
##                    res = calculeRot(mat)
##                    if res: m = dot(mat,mMatPrec)
##                    if not res:
##                        res = [calculeRot(dot(-1,mat)),"*",-1]
##                    print res
##                print
##            print "---------------------------------------"
            QDirPos = []
            for lmat in QDir:
                lmatpos=[]
                for mat in lmat:
                    if det(mat)>0: lmatpos.append(mat)
                QDirPos.append(lmatpos)
            print "--------------------------------------"
            listSubGroups = FindSubGroups(QDirPos)
            print len(listSubGroups), "sous-groupes trouves dans les matrices positives"
            for subg in listSubGroups:
                print
                for mat in subg: print calculeRot(mat)
            print "--------------------------------------"
            #for v in listVect: print v
            
            #ConjugateSubGroups(QDir)
            SubGroup= listSubGroups[0]
            listOp = []
            for g in SubGroup:
                listOp.append(dot(g,mMatPrec))
       
            showTransition_Tk(root2,choiceOR,listOp)

        
    #----------------------------------------------------------------------------------------------------
    def DrawPloleFig():
            global modeFig
            choiceOR= str(root2.fields[0].get ())
            fileMat     = root2.fields[2].get ()
            filePrec        = root2.fields[3].get ()
            crystM = importCryst(fileMat,phaseDirectory)
            crystP = importCryst(filePrec,phaseDirectory)    
            GMat,  SMat, symMat = crystM.metricTensDir,crystM.structTensDir,crystM.sym()
            GPrec, SPrec, symPrec   = crystP.metricTensDir,crystP.structTensDir,crystP.sym()

            Directory = disque+"\\"+versionP+"\\arpge\\Crystallography\\Transitions\\" + choiceOR
            try: os.chdir (Directory)
            except: os.chdir (disque+"\\"+versionP+"\\arpge\\Crystallography\\Transitions\\Current")
            if choiceOR!="Sigma3":
                
##                f = open('CC_Groupoid_AllInfos.txt','r')
##                choiceOR,mMatPrec,QDir,listConfig,ListDir,GrDir,TableCompOp = pickle.load(f)
##                f.close()

                f = open('CC_PhaseTransition.txt','r')
                mMatPrec, QDir, QDual,TabVar,listOpTarget,Tab = pickle.load(f) 
                f.close()

                print "mMatPrec"
                print mMatPrec
                print "===="
            
            if choiceOR=="Sigma3":
                f = open('CC_listTwin.txt','r')
                lopTwinRat=pickle.load(f)
                f.close()
                listOp = []
                size=400
                i=0
                for lop in lopTwinRat:
                    i=i+1
                    lopfloat=[]
                    sigma = 3**i
                    for op,ch in lop:
                        opfloat = dot(1./sigma,op)
                        lopfloat.append(opfloat)
                    listOp.append(lopfloat)
                    if i>OrderTwin: break
                print OrderTwin
                print "LISTE DES OPERATEURS"
                for op in listOp:
                    print "-----------------------"
                    print op
                
                PolFigVariants(root2,size,listOp,GPrec, SPrec, symPrec, GMat, SMat, symMat,choiceOR)
            else:
                size=400
                listOp = []
                for lmat in QDir:
                    listOp.append(dot(lmat[0],mMatPrec))
                PolFigVariants(root2,size,[listOp],GPrec, SPrec, symPrec, GMat, SMat, symMat,choiceOR)
       
                
    def tkCalculation ():
            global phaseDirectory
            global fileMat
            global filePrec
            global zoneAxisMat
            global p1Prec 
            global p1Mat
            global p2Prec 
            global p2Mat
            global d1Prec
            global d1Mat
            global d2Prec
            global d2Mat
            global precAxisTolerance
            global imageSize
            global diameterMat
            global diameterPrec
            global colorFlag
            global gradientFlag
            global borderFlag
            global indiceFlag
            global variantFlag
            global Sauv
            global Data
            global OrderTwin
            global choiceOR

            choiceOR            = root2.fields[0].get ()
            phaseDirectory  = root2.fields[1].get ()
            fileMat     = root2.fields[2].get ()
            filePrec        = root2.fields[3].get ()
            p1p         = root2.fields[4].get ()
            p1m         = root2.fields[5].get ()
            p2p         = root2.fields[6].get ()
            p2m         = root2.fields[7].get ()
            d1p         = root2.fields[8].get ()
            d1m         = root2.fields[9].get ()
            d2p         = root2.fields[10].get ()
            d2m         = root2.fields[11].get ()
            precAxisTolerance   = float (root2.fields[12].get ())   
            colorFlag       = root2.tkColorFlag.get ()
            gradientFlag    = root2.tkGradientFlag.get ()
            borderFlag      = root2.tkBorderFlag.get ()
            indiceFlag      = root2.tkIndiceFlag.get ()
            doubleDiffFlag  = root2.tkDDFlag.get ()
            Sauv                = root2.tkSauv.get ()
            DataFlag            = root2.tkDataFlag.get ()
            OrderTwin = string.atoi(entry_order.get())
            
            p1Mat, p1Prec = [], []
            if p1p and p1m: 
                    p1p = string.split (p1p, ',')
                    p1Prec = [string.atof(p1p[0]), string.atof(p1p[1]), string.atof(p1p[2])]
                    p1m = string.split (p1m, ',')
                    p1Mat = [string.atof(p1m[0]), string.atof(p1m[1]), string.atof(p1m[2])]

            p2Mat, p2Prec = [], []
            if p2p and p2m: 
                    p2p = string.split (p2p, ',')
                    p2Prec = [string.atof(p2p[0]), string.atof(p2p[1]), string.atof(p2p[2])]
                    p2m = string.split (p2m, ',')
                    p2Mat = [string.atof(p2m[0]), string.atof(p2m[1]), string.atof(p2m[2])]
            
            d1Mat, d1Prec = [], []
            if d1p and d1m:
                    d1p = string.split (d1p, ',')
                    d1Prec = [string.atof(d1p[0]), string.atof(d1p[1]), string.atof(d1p[2])]
                    d1m = string.split (d1m, ',')
                    d1Mat = [string.atof(d1m[0]), string.atof(d1m[1]), string.atof(d1m[2])]
            
            d2Mat, d2Prec = [], []
            if d2p and d2m:
                    d2p = string.split (d2p, ',')
                    d2Prec = [string.atof(d2p[0]), string.atof(d2p[1]), string.atof(d2p[2])]
                    d2m = string.split (d2m, ',')
                    d2Mat = [string.atof(d2m[0]), string.atof(d2m[1]), string.atof(d2m[2])]
            
            if (    ((d2p or d2m) and (p1p or p1m))
                 or ((d1p or d1m) and (p2p or p2m))
                 or ((d2p or d2m) and (p2p or p2m))):
                    self.infobox ('Error !',
                                  'You have to choose either p1 &amp; p2 or p1 &amp; d1 or ' \
                                  + 'd1 &amp; d2 for the matrix and the precipitate.',
                                  bitmap = '', strings = ('OK',))
                    
            if Sauv:
                directory = transitionDirectory+"\\"+"ListAllTransitions"
                nametrans = choiceOR+".trans"
                r=True
                if nametrans in os.listdir(directory):
                    r = tkMessageBox.askyesno('Warning','Overwrite the existing file'+ nametrans + ' ?')
                if r: 
                    os.chdir (directory)
                    f = open(choiceOR+".trans",'w')
                    l = [p1Prec,p1Mat,p2Prec,p2Mat,d1Prec,d1Mat,d2Prec,d2Mat]
                    info =[choiceOR,fileMat,filePrec]
                    for e in l:
                        newe = map(int,e)
                        info.append(newe)
                    pickle.dump(info,f)
                    f.close()
                    try: os.mkdir(transitionDirectory+"\\"+choiceOR)
                    except: pass
                else: sys.exit()
                
            doCalculation (OrderTwin)
            
                

    makeWidgets(root2)

# ==============================================================================

if __name__ == '__main__':

    root=Tk()
    bar  = Menu(root)
    root.title("GenOVa 2.0 (C. Cayron, 2016)")
    
    choix = "Sigma3"
    choix = "Burgers"
    choix = "Fe-NW"

    frame = Frame(root,borderwidth=2,width=1000,height = 500)
    frame["relief"]=RAISED
    frame.grid(row=0,column=0)


    TkVariants(frame)
    root.mainloop()

# ==============================================================================


