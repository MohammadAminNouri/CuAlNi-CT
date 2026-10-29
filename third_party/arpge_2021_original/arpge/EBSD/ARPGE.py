# author: Cyril CAYRON

# Sept 2021
# C. Cayron
# programme de reconstruction de grains parents par exploitation de cartos EBSD
# 1ers developpements faits a l EPFL pdt these 1996-2000
# et base du programme de C. Cayron et B. Artaud (stage encadre par C. Cayron, 2004)
# partie de reconstruction modifiee en profondeur Novembre 2006 - Mars 2007
# partie sigma3n ecrite fin 2008, debut 2009
# partie packets pour martensite des aciers fin 2013-debut 2014
# partie KAM, GOS, figures de poles en densite de points en 2015
# bug sur le codage ang/axe des operateurs corrige dans GenOVa en Oct 2016
# bug sur les figures de poles des non-cubiques corrige dans PoleFig en Oct 2016
# trace des plans dhabitat en Janv 2017
# correction de CC_Eulers pour les monocliniques et modif du tenseur de structure pour hexagonaux, Oct 2017
# cette correction portait un bug qui a ete corrige en Nov 2018
# retained parent code par TabPix [5] = bc = 300 et TabPix[4] = ind parent grain negatif Nov 2017
# IMPORTANT: LES FICHIERS DE BASE SONT GENERES PAR le programme CC_GenOVa.py

import sys, os , string
import time

from math import *
from numpy import *
from numpy.linalg import inv, eig, det

from Tkinter import *
from tkFileDialog import *
import tkMessageBox
import pickle
import copy
import gzip
import random
from PIL import Image, ImageTk, ImageDraw, ImageFont, ImageGrab

try: import winsound
except: pass

from Canvas import Rectangle, Oval, Arc, Window
#from colormap           import SetupColormap0

from arpge.Crystallography.CC_Eulers          import EulersToMat, MatToEulers

from arpge.Algebra.CC_AlgebraicTools     import IntegriseAxe, cross, length, normalize, calculeRot, calculeMatRot

from arpge.Crystallography.CC_Crystals        import lengthCryst, normalizeCryst, angle, importCryst, Disorientation, SimpleAxe, vectDir

from arpge.Graphics.CC_Histogram       import Histogram
from arpge.Graphics.CC_PlotPoleFigure  import PolFigEulers, PolFigDirections, IPFaxes, IPFCoord
from arpge.Graphics.CC_TRDGraph        import TwinTreeListOp

from arpge.Utils.Files import getFullPath, getResultsPath

#sys.setrecursionlimit(10000) # sera utile pour la reconnaissance des grains filles, valeur par defaut de Python = 1000
RecursionLim = sys.getrecursionlimit() - 100

EBSDDirectory = getResultsPath()

# overwrite arpge.cfg
v = sys.version[:3]
versionP = "Python" + str(reduce(lambda x,y:x+y, v.split(".")))
try:
    N = "c:/"+versionP+"/arpge"
    os.chdir(N)
    disque = "c:"
except:
    try:
        N = "d:/"+versionP+"/arpge"
        os.chdir(N)
        disque = "d:"
    except:
        print "error in lauching ARPGE: arpge folder not found"
t = "[Results]" + "\n" + "path="+disque+"/"+versionP+"/ElectronMicroscopy_Data"
fname = disque+"/"+versionP+"/arpge" + "/arpge.cfg"
f = open(fname, 'w')
f.write(t)
f.close()
EBSDDirectory = t

directory = getFullPath("Crystallography")
filename = 'PointGroups_EquivEulers.data'
f = open(os.path.join(directory, filename), 'r')
DictPGFEulers = pickle.load(f)
f.close()

#------------------------------------
#File text required to export the data
os.chdir (disque+"/"+versionP+"/arpge/EBSD")
bufile = open("TSL_Fe.txt")
TSL_FeGamma = bufile.read()
bufile.close()
bufile = open("TSL_Ti.txt")
TSL_TiBeta = bufile.read()
bufile.close()
bufile = open("TSL_ZrO2.txt")
TSL_ZrO2_cubic = bufile.read()
bufile.close()
HKL_FeGamma = "Phases	1\n"+"3.6599;3.6599;3.6599	90;90;90	Iron fcc	11	225	3803863129_5.0.0.1277	1619986839\n"
HKL_TiBeta =  "Phases	1\n"+"3.3100;3.3100;3.3100	90;90;90	Ti bcc	        11	229	3803863129_5.0.0.1277	1619986839\n"
HKL_ZrO2_cubic =  "Phases	1\n"+"5.2000;5.2000;5.2000	90;90;90	ZrO2 cubic	        11	225	3803863129_5.0.0.1277	1619986839\n"
t_HKL = "Phase	X	Y	Bands	Error	Euler1	Euler2	Euler3	MAD	BC	BS\n"
HKL_FeGamma = HKL_FeGamma + t_HKL
HKL_TiBeta =HKL_TiBeta + t_HKL
#------------------------------------

COLORS_POL = ['black', 'grey', 'red', 'yellow', 'blue', 'dark green', 'light green', 'deep pink', 'hot pink', 'dark orange',
              'salmon']

COLORS_AMB = ['black', 'grey', 'red', 'yellow', 'blue', 'green', 'pink', 'orange']

COLORS_OP = ['black', 'grey', 'red', 'yellow', 'blue', 'green', 'pink', 'orange', 'salmon', 'violet', 'brown', 'plum', 'chocolate', 'coral', 'navy', 'tomato', 'linen',
             'cyan', 'gold', 'khaki', 'tan', 'rosy brown', 'ivory', 'moccasin','dark green']

COLORS = COLORS_AMB

ListPlanesCubic = [[1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 1, 2]]
ListPlanesHex = [[1, 0, 0], [1, 1, 0], [1, 0, 1], [0, 0, 1]]

global Longueur
global Nbpixx
global Nbpixy
global HMAX, WMAX # screen size
global TabPix # 0 E1,1 E2, 2 E3, 3 graind, 4 grainp, 5 BC # daughter pixels, imported from HKL or TSL NEW Nov 2017 BC = 300 == Retained parent
global TabPixP # calculated parent pixels 0 E1,1 E2, 2 E3
global DilatationRecurs # for dilatation around daughter pixels by recursion
global TabGrains
global FileName
global AngTolGrain
global AngTolParentGrowth     # for parent grain growth
global AngTolParentNucl # for parent grain nucleation
global AngTolGBStraight # for the identification of straight GBs
global MinGrainSize # minimum grain size in pixel for the parent grain reconstruction
global MinRetainedSize # minimum grain size in pixel for the retained parent grain identification
global Scale    #echelle pour cartes EBSD
global OptShowA  #option de carte EBSD euler1,2,3, and packets
global OptShowB  #option de carte EBSD pixel, dauhgter grains, parents grains
global OptShowC # Line, Rectangle pour faire la disorientation le long dune line ou definir un rectangle de selection
global OptShowN #montre ou pas les numeros des grains
global OptShowRet # 0 = "Only the reconstructed parent grains are shown",  1 = "Only retained parent grains are shown" , 2 = "Reconstructed and retained parent grains are shown"
global ComptNucl # 0 avant une premiere reconstruction3 (avec triplets ou quadruplets), ensuite incremente de 1 a chaque reconstruction
global ComptGrowth # 0 avant une premiere reconstruction (avec triplets ou quadruplets), ensuite incremente de 1 a chaque reconstruction
global FlagRec #1 si 1ere reconstruction, 0 sinon
global currentindgrp #indice du grain parent en cours
global TabGrainsRec, TabOrient  #pour la reconstruction
global ListIndGrdGrowth
global listNuclei #pour la reconstruction
global iparent  #pour la reconstruction
global GRID # for TSL files
global ReLoadFlag  # 1 required for reloading the file

global image, imageTk, imageCV, canvas

global listOpDisor #liste des operateurs (tous avec les symmeties!)
global listOp180 #liste des operateurs a 180 degres
global Transition # string
global QuotientDir # ensemble quotient = ensemble des variants Dir = direct transition, Dual = inv transition
global QuotientDual # ensemble quotient = ensemble des variants Dir = direct transition, Dual = inv transition
global PtoD # transformation matrix from a parent basis to a basis of the daughter variant 1
global TSD, TMD #tenseur de structure et tenseur metrique de la phase fille
global TSP, TMP #tenseur de structure et tenseur metrique de la phase parente
global GroupeRotD # groupe ponctuel de la phase fille
global GroupeRotP # groupe ponctuel de la phase parente
global listOpVar #liste des operateurs sous forme [source,target]
global TabCompoOp  # table de composition du groupoide
global listOpTarget # liste des operateurs sous forme {[i]} represantant {[v1vi]}
global listOpSource # liste des operateurs sous forme {[i]} represantant {[viv1]}
global FEulersP #[fts de reduction des angls d Eulers, liste des symetries residuelles (pour les cubiques)]
global FEulersD
global IPFaxesD, IPFaxesP # [{3 axes}, [CrystIPF->Cryst], anga angb, angc, surfsphere]
global Units #microns or nm
global DictTwinNom #Dictionnaire pour savoir le nom dun operateur a partir du chemin
global DictTwinPos #Dictionnaire pour savoir le numero dun operateur a partir du nom
global TRDTreeDone # to know if the TRDTree procedure has been done or not
global NextStep # help the user to different steps of the software
global PercentRec #percentage of the grains that have been reconstructed
global PercentRecLimit # threshold for automatic reconstruction
global PercentOrient # percentage of the grains that have been orientated
global NbMinGrd # min number of daughter grains in parent grains for the automatic reconstruction
global AngTolGrainStep # step for the automatic reconstruction
global AngTolParentGrowthStep     # step for the automatic reconstruction
global AngTolParentNuclStep # step for the automatic reconstruction
global AngTolParentNuclLimit # max tolerance angle for the nucleation step
global AngTolParentGrowthLimit # max tolerance angle for the growth step
global FlagSupPix # flag for the automatic reconstruction (dilate or not)
global FlagAutoSave # flag for the automatic reconstruction (save the results or not)
global FlagPixP # to alternate showpix between exp. daughter and calc. parent pixels
global ListPlanes # exhaustive list of planes compared to straight GB
global ListGBPlanes # list of planes compared to straight GB defined by the user
global ListOpGBLines # list of GB Lines for all the GB calculated in DrawGBOp
global ListOpGBPercL # list of repartion  (percentages) of special GBs [ random, op 0 etc.] in linear perc.
global listMatDtoP # list of matrices from Daughter to Possible Parents
global listQPos # pour calculer les orientations des variants a partir de celle du grain parent
global ShapeFact # for reshaping the frontiers of parent grains
global IndexQual #index quality for the variant indexation
global SbSize # size of the scalebar
global NumSizeBig # size of Parent Grain Num
global NumSizeSmall # size of the Daughter Grains Num
global TabPixTriangle # uniquement pour transitions NW, Pitsch ou KS ou G: sert au codage dans le triangle P-KS-NW
global frameMenuCV, ComptTriangle
#global ButtonBain, ButtonCPD, ButtonCPP, ButtonBurst, ButtonTri # for P-NW-KS shown only for the ORs NW KS or GT. ComptTriangle = 0 or 1
global ResTri # chain, result of the Triangle Pitsch, KS, NW
global Entete # The fist lines at begining of the EBSD data file. Will be also used in the exported file.
global iPhase # line of the phase in the header in the ctf file



SourceFile = '' # can be HKL (.ctf), TSL (.ang), ASTAR (.ang) and soon Bruker
TransLoadMode = 0 # savoir si reouvre une transition
Longueur = 0
Nbpixx = 200
Nbpixy = 200
Xmax = 100
Ymax = 100
XStep = 1
YStep = 1
ScaleBar = 1 #en pixels
DilatationRecurs = 0
SBSize = 10 
NumSizeBig = 15 
NumSizeSmall = 10 
TabPix = []
TimeStart = 0
AngTolGrain = 5. #degree
AngTolParentNucl = 3. #degree
AngTolParentGrowth = 6. #degree
AngTolParentNuclLimit = 15 # degree
AngTolParentGrowthLimit = 20 # degree
AngTolGBStraight = 10. #degree
MinGrainSize = 8 #nb de pixels pour ignorer un grain
Scale = 1.
OptShowA = "euler1"
OptShowB = "pixels"
OptShowC = "Line"
OptShowN = 0
OptShowRet = 0
Transition = "?"
Units =  u"\u03BCm"
TabGrD = [] #liste des grains sous forme de listes d indexes de pixels
TabGrains = [] #daughter grains
TabParentGrains = [] # reconstructed parent grains
TabRetainedGrains = [] #retained parent grains TabPix[i][5] ==300 and TabPix[i][4] <0 
ComptNucl = 0
ComptGrowth = 0
FlagRec = 1 # 0 reprend la reconstruction au moment ou il l a laisse, si 1 reprend tout a zero
currentindgrp = -1
TabGrainsRec = []
TabOrient = []
listNuclei = []
listIndGrdGrowth = []
iparent = 0
OrderTwin = 3 # ordre max de maclage explore pour reconstruction des TRD par recherche entre grains voisins
GRID = "Square"
#GRID = "Hexagon"
DictTwinNom = {}
DictTwinPos = {}
TRDTreeDone = 0
NextStep = 0
PercentRec = 0
PercentRecLimit = 99.9 #percents
PercentOrient = 0
NbMinGrd = 5 #min number of daughter grains in the reconstructed parent grains
AngTolGrainStep = 1 # step for the automatic reconstruction
AngTolParentGrowthStep = 2    # step for the automatic reconstruction
AngTolParentNuclStep = 2 # step for the automatic reconstruction
ListPlanes = [] # valeur par defaut dans LoadTrans, et sera defini dans Parameters
ListGBPlanes = []
ListOpGBLines = []
ListOpGBPercL = []
listOpDisor = []
listOp180 = []
listMatDtoP = []
ShapeFact = 0.7
IndexQual = 0
StProg = 50 # nb of steps in the Progress bar
FlagPixP = 0
TabPixP = []
TabPixDef = [] # table as TabPix, with anglar deformation inside
defmin, defmax = 0, 0 # KAM = angular deformation min and max (Deg.) 
KAMmean = 0  # KAM average on all the pixel of daughter map
gosmin,gosmax = 0,0 # Grain Orientation Spreading calculated for each daughter grain
GOSmean = 0 # GOS average on all the daughter grains 
ComptTriangle = 0
TabPixTriangle = []
ResTri = ''
modeIPF = 1 #0 = X = RD, 1 = Y = TD, 2 = Z = ND
Entete = ""
iPhase = 0
ReLoadFlag = 0
MinRetainedSize = MinGrainSize


PhaseDirectory	 = getFullPath(os.path.join("Crystallography", "PhasesCRYST"))
TransitionDirectory = getFullPath(os.path.join("Crystallography", "Transitions"))
EBSDDirectory = getResultsPath()
print "Directory for EBSD : ",EBSDDirectory 
os.chdir (EBSDDirectory)
FileName = ""

# Channel5 HKL ctf
# col = {'phase':0, 'x':1, 'y':2, 'Bands':3, 'Error':4, 'euler1':5, 'euler2':6, 'euler3':7, 'MAD':8, 'BC':9, 'BS':10}
# OIM TSL ang
# 0.39122   1.82022   0.17254      0.00000      0.00000   67.4  0.917  0      1  0.712

NbColors = 255
#cmap = SetupColormap0(NbColors)

Identity = array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])

DEGREE = 180. / math.pi

listMatS3 =[array([[ 0.66666667, -0.66666667,  0.33333333],
                   [ 0.33333333,  0.66666667,  0.66666667],
                   [-0.66666667, -0.33333333,  0.66666667]]),
            array([[ 0.66666667,  0.66666667,  0.33333333],
                   [-0.33333333,  0.66666667, -0.66666667],
                   [-0.66666667,  0.33333333,  0.66666667]]),
            array([[ 0.66666667, -0.66666667, -0.33333333],
                   [ 0.33333333,  0.66666667, -0.66666667],
                   [ 0.66666667,  0.33333333,  0.66666667]]),
            array([[ 0.66666667,  0.33333333, -0.66666667],
                   [-0.66666667,  0.66666667, -0.33333333],
                   [ 0.33333333,  0.66666667,  0.66666667]]),
            array([[ 0.66666667, -0.33333333, -0.66666667],
                   [ 0.66666667,  0.66666667,  0.33333333],
                   [ 0.33333333, -0.66666667,  0.66666667]]),
            array([[ 0.66666667,  0.33333333,  0.66666667],
                   [-0.66666667,  0.66666667,  0.33333333],
                   [-0.33333333, -0.66666667,  0.66666667]]),
            array([[ 0.66666667, -0.33333333,  0.66666667],
                   [ 0.66666667,  0.66666667, -0.33333333],
                   [-0.33333333,  0.66666667,  0.66666667]]),
            array([[ 0.66666667,  0.66666667, -0.33333333],
                   [-0.33333333,  0.66666667,  0.66666667],
                   [ 0.66666667, -0.33333333,  0.66666667]])]

#----------------------- FOR TWINNING -----------------------------
# matrices used for twinning 

T0 = array([[ 1, -2, -2], [-2, 1, -2], [-2, -2, 1]])    # mirroir 1 1 1

T1 = array([[1, -2, -2], [2, 2, -1], [2, -1, 2]])               # rotation 70.5   0 -1 1
m1 = array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])                   # + mirroir
T1 = dot(T1, m1)


T2 = array([[2, 2, -1], [-2, 1, -2], [-1, 2, 2]])               # rotation 70.5   -1 0 1
m2 = array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])                   # + mirroir
T2 = dot(T2, m2)


T3 = array([[2, -1, 2], [-1, 2, 2], [-2, -2, 1]])               # rotation 70.5   1 -1 0
m3 = array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])                   #+ mirroir
T3 = dot(T3, m3)

T0 = dot(-1. / 3, T0)
T1 = dot(-1. / 3, T1)
T2 = dot(-1. / 3, T2)
T3 = dot(-1. / 3, T3)

ListTwinRef = [T0, T1, T2, T3]

#-------------------------------------------------------------------------

# javais cree un classe pixel mais elle prend trop de place memoire pour coder les cartos EBSD
# donc chaque pixel est sous forme d une liste
# pix = euler1,euler2,euler3,graind, grainp

#================================================================================================

class Grain:
    def __init__(self, phase, x, y, euler1, euler2, euler3, graind, variant, grainp, listpix, listpixfr, listgrvois, confind = 30, kam = 0, gos = 0):
        #graind = daughter grain
        #grainp = parent grain
        self.phase = phase
        self.x = x
        self.y = y
        self.euler1 = euler1            # valeurs moyennes des pixels
        self.euler2 = euler2
        self.euler3 = euler3
        self.graind = graind            # numero du graind
        self.variant = variant          # numero du variant (list of possibilities)
        self.grainp = grainp
        self.listpix = listpix          # liste des pixels contenus dans le grain (liste d indices)
        self.listpixfr = listpixfr      # liste des pixels de frontiere de grain (liste d indices)
        self.listgrvois = listgrvois    # liste des grains voisins (liste d indices)
        self.confind = confind          # confidence index (only for parent grain) 0=good, 30 = bad
        self.kam = kam
        self.gos = gos

    def toList(self):
        phase = self.phase
        x = self.x
        y = self.y
        euler1 = self.euler1
        euler2 = self.euler2
        euler3 = self.euler3
        graind = self.graind
        variant = self.variant
        grainp = self.grainp
        listpix = self.listpix
        listgrvois = self.listgrvois
        confind = self.confind
        kam = self.kam
        gos = self.gos
        return phase, x, y, euler1, euler2, euler3, graind, variant, grainp, listgrvois, kam, gos #, listpix[:20]

#================================================================================================

class cloneGrain(Grain):
    def __init__(self, gr):
        self.phase = gr.phase
        self.x = gr.x
        self.y = gr.y
        self.euler1 = gr.euler1
        self.euler2 = gr.euler2
        self.euler3 = gr.euler3
        self.graind = gr.graind
        self.variant = gr.variant
        self.grainp = gr.grainp
        self.listpix = gr.listpix
        self.listpixfr = gr.listpixfr
        self.listgrvois = gr.listgrvois
        self.confind = gr.confind
        self.kam = gr.kam
        self.gos = gr.gos

#===================================================================================================

def cut(nb, n=3):
    if int(nb)==nb: return str(nb)
    else:
        t = str(round(nb,12))
        t = t[:t.find(".") + n]
        return t

#================================================================================================

def findfile(filename, racine):
    
    # cherche le repertoire ou se situe le fichier
    global res
    
    res = '?'
    def lookall(bidon, chemin, liste_noms):
        global res
        for nom in liste_noms:
            if filename in os.path.join(chemin, nom):
                res = chemin
                break
    os.path.walk(racine, lookall, None)
    return res

#================================================================================================

def PosXY(pos):
    #pix sous forme de liste euler1, euler2,euler3,graind,grainp
    
    if GRID != "Hexagon":
        i = pos / Nbpixx
        j = pos - (i * Nbpixx)
        x,y = j * XStep, i * YStep
        # x,y = j * XStep, i * XStep*(sqrt(3)/2) #avec TSL hexgagon
    else:
        i = pos / Nbpixx
        if i%2==1:
            j = pos-(i * Nbpixx)+1./2
        else:
            j = pos - (i * Nbpixx)
        x,y = j * XStep, i * YStep*(sqrt(3)/2)
    return [x,y]

#=================================================================================================

def intersec(l1, l2):
    # pas la peine de passer par les set pour les petites listes

    res = []
    if not l1 or not l2:
        return res
    else:
        for i in l1:
            if i in l2: res.append(i)
    return res

#=================================================================================================

def intersec2(l1, l2):
    # renvoie aussi le residu de l1 qui nappartient pas a l2
    # reserve pour les listes <1000 ce qui est le cas des listpixfr des grd
    # il faudrait utiliser les sets si listes >1000

    resi,resr = [[],[]]
    if not l1 or not l2:
        return [[],[]]
    else:
        for i in l1:
            if i in l2: resi.append(i)
            else: resr.append(i)
    return [resi,resr]

#=================================================================================================

def renum(n,listr):
    # renumerote un nombre entier x en fonction d une liste entiers listr qui seront effaces
    res = 0
    mini, maxi = min(listr), max(listr)
    if n < mini: res = n
    elif n> maxi: res = n-len(listr)
    else:
        ii=0
        for r in listr:
            if r==n:
                res =0
                break
            elif n<r:
                res = n-ii #ii = position of n in the list = nb of grd removed
                break
            ii+=1
    return res
    
#=================================================================================================

def listpixij(listpix):
    
    lij = []
    for pix in  listpix:
        i = pix / Nbpixx
        j = pix - Nbpixx * i
        lij.append([i, j])
    return lij

#=================================================================================================

def intersecpix(l1, l2):

    res = []
    if not l1 or not l2:
        return res
    else:
        for pix1 in l1:
            for pixv1 in Lvoisinspix4(pix1):
                if pixv1 in l2: res.append(pix1)
    return res

#===================================================================================================

def compoOp(ij, ik):
    # calcule la composition de deux operateurs Oij et Oik = Oij-1 Oik = {Ojk} grace aux variants
    
    listOjk = []
    listjk = []
    for  Oij in listOpVar[ij]:
        for Oik in listOpVar[ik]:
            if Oij[0] == Oik[0]:
                listOjk.append([Oij[1], Oik[1]])
    for Ojk in listOjk:
        for i in range(len(listOpVar)):
            listO = listOpVar[i]
            flag = 0
            for O in listO:
                if O == Ojk:
                    if i not in listjk: listjk.append(i)
                    flag = 1
                    break
            if flag: break
    return listjk

#===================================================================================================

def compoOpVar1(ij, ik):
    # calcule la composition de deux operateurs O1j et O1k grace aux variants
    
    listOjk = []
    listResjk = []
    for  Oij in listOpVar[ij]:
        if Oij[0] == 1:
            for Oik in listOpVar[ik]:
                if Oik[0] == 1:
                    listOjk.append([Oij[1], Oik[1]])
    for Ojk in listOjk:
        for i in range(len(listOpVar)):
            listO = listOpVar[i]
            flag = 0
            for O in listO:
                if O == Ojk:
                    listResjk.append([Ojk, i])
                    flag = 1
                    break
                if flag: break
    return listResjk
#===================================================================================================

def CalcTabCompoOp():
    
    global TabCompoOp

    TabCompoOp = [[[] for i in range(len(listOpVar))] for j in range(len(listOpVar))]
    for i in range(len(listOpVar)):
        for j in range(len(listOpVar)):
            TabCompoOp[i][j] = compoOp(i, j)

    return
# ==============================================================================

def CodeTwinToMatrix(ch):
    # JUST for multiple twinning: transform the code into a matrix
    
    orvar = Identity
    for i in ch:
        orvar = dot(orvar, ListTwinRef[i])
    return orvar

#===================================================================================================

def stringify(ch):
    # change un chemin twin sous forme de liste en une chaine
    
    s = ''
    for el in ch: s = s + str(el)
    return s

#===================================================================================================

def GenerateEquiv(listg, group):
    
    newlisteqg = []
    for g in listg:
        lng = []
        for sym in group:
            ng = dot(sym, array(g))
            ngminus = dot(-1, ng).tolist() # il faudrait retirer ca pour les non centrosymetriques
            ng = ng.tolist()
            if ng not in lng and ngminus not in lng: lng.append(ng)
        newlisteqg.append(lng)
    return newlisteqg

#===================================================================================================

def test (mat, Tolang): #mat matrice de disorientation
    
    # compare la disorientation experimentale aux disorientations theoriques

    if len(listOp180) > 0:
        angle, axe = calculeRot(mat)
        if abs(abs(angle) - 180) < 3: # les rotations a 180 degrees ne peuvent pas etre facilement reconnues par leurs matrices=> cette methode brutale
            return copy.deepcopy(listOp180)

    #invmat = inv(mat) en fait pour eviter de prendre linverse on compare dans le tableau de composition la transpose du tableau
    listangmin = []
    for indop in range(len(listOpDisor)):
        deltaangmin = 360
        lmatop = listOpDisor[indop][0][2] # [0] = angmin = disorientation, [2] = listmat de disorientation
        for matop in lmatop:
            m = dot(mat, matop)
            tr = trace(m)
            if tr < -1: tr = -1
            if tr > 3: tr = 3
            ad = abs(arccos((tr - 1) / 2)) * DEGREE
            if (ad < deltaangmin):
                deltaangmin = ad
        if deltaangmin < Tolang: listangmin.append([deltaangmin, indop])

    listangmin.sort()
    l = []
    for i in range(len(listangmin)):
        l.append (listangmin[i][1])

    return l

#===================================================================================================

def testS3nInList(mat, listnomop, angtol):
    
    #listOpDisor = [] # liste des matrices op [angmin,axe, lmat]
    #listTwinNom = [] # liste des noms des operateurs [nom,lch] = meme taille que listOpDisor
    res = -1

    invmat = inv(mat)
    for nomop  in listnomop:
##        i=0
##        for nom,lch in listTwinNom: # UTILISER DICTTWIN voir TRDgraph.py
##            #print "nom=",nom
##            if nomop==nom: break
##            i+=1
        i = DictTwinPos[nomop]
        if i < len(listOpDisor):
            lmatop = listOpDisor[i][0][2] # on pourrait aller a lordre 8 mais dans loadtransitiontwin je me suis arrete a lordred 6
            deltaangmin = 360
            for matop in lmatop:
                m = dot(invmat, matop)
                tr = trace(m)
                ad = abs(arccos((tr - 1) / 2)) * DEGREE
                if (ad < deltaangmin):
                    deltaangmin = ad
            if deltaangmin < angtol:
                res = i
                break
        else: res = -1
    return res

#===================================================================================================

def testS3n (mat, Tolang, order=1):

    #mat matrice de disorientation
    # compare la disorientation experimentale aux disorientations theoriques
    # copier coller de test mais en coupant les operateurs a Sigma3n 
    # n<= listTwinNum[order+1]

    invmat = inv(mat)
    res = -1
    for indop in range(len(listOpDisor[:listTwinNum[order + 1]])):
        deltaangmin = 360
        lmatop = listOpDisor[indop][0][2] # [0] = angmin = disorientation, [2] = listmat de disorientation
        for matop in lmatop:
            m = dot(invmat, matop)
            tr = trace(m)
            ad = abs(arccos((tr - 1) / 2)) * DEGREE
            if (ad < deltaangmin):
                deltaangmin = ad
        if deltaangmin < Tolang:
            res = indop
            break
            #listangmin.append([deltaangmin,indop])

    return res

#===================================================================================================

def testS3 (mat, Tolang):

    #mat matrice de disorientation a comparer aux matrices Sigma3
    # compare la disorientation experimentale aux disorientations theoriques

    res = 0
    invmat = inv(mat)
    deltaangmin = 360
    #lmatop = listOpDisor[1][0][2] # le 1 pour Sigma3, (2 = sigma9 etc), [0] = angmin = disorientation, [2] = listmat de disorientation
    lmatop = listMatS3 # = listOpDisor[1][0][2] for Sigma3 transitions
    for matop in lmatop:
        m = dot(invmat, matop)
        tr = trace(m)
        ad = abs(arccos((tr - 1) / 2)) * DEGREE
        if (ad < deltaangmin):
            deltaangmin = ad
    if deltaangmin < Tolang: res = 1

    return res

#===================================================================================================

def testId (mat, Tolang):

    #m matrice de disorientation

    res = 0
    tr = trace(mat)
    ad = abs(arccos((tr - 1) / 2)) * DEGREE
    if ad < Tolang: res = 1

    return res

#===================================================================================================

def testIdang (mat):

    #m matrice de disorientation

    tr = trace(mat)
    if tr >3: tr = 3
    elif tr <-1: tr = -1
    ad = abs(arccos((tr - 1) / 2)) * DEGREE

    return ad
#===================================================================================================

def FonctSource(op, varT):

    #donne les numeros des variants source vars dont (vars,vart)=op

    res = []
    listvar = listOpVar[op]
    for st in listvar:
        s, t = st
        if t == varT: res.append(s)
    return res

#===================================================================================================

def FonctTarget(op, varS):

    #donne les numeros des variants target vart dont (vars,vart)=op

    res = []
    listvar = listOpVar[op]
    for st in listvar:
        s, t = st
        if s == varS: res.append(t)
    return res

#====================================================================================================

def VariantProgram():

    try:
        from arpge.Crystallography.CC_GenOVa import TkVariants
        TkVariants(root)
    except:
        EtatRun.set("\n"+"You need GenOVa. Please contact C. Cayron."+"\n")
        LabelRun["fg"] = "red"
        frameFile.update()

#====================================================================================================

def Burgers():

    global Transition, TransLoadMode, ListPlanes, RecMethod

    Transition = "Burgers"
    ListPlanes = ListPlanesHex
    TransLoadMode = 1
    #RecMethod.set(1) # reconstruction with quadruplets
    LoadTransitionFiles()

#====================================================================================================

def NW():

    global Transition, TransLoadMode, ListPlanes

    Transition = "Fe-NW"
    ListPlanes = ListPlanesCubic
    TransLoadMode = 1
    LoadTransitionFiles()
#====================================================================================================

def KS():

    global Transition, TransLoadMode, ListPlanes

    Transition = "Fe-KS"
    ListPlanes = ListPlanesCubic
    TransLoadMode = 1
    LoadTransitionFiles()
    
#====================================================================================================

def GT():

    global Transition, TransLoadMode, ListPlanes

    Transition = "Fe-GT"
    ListPlanes = ListPlanesCubic
    TransLoadMode = 1
    LoadTransitionFiles()

#====================================================================================================

def ZirconiaCM():

    global Transition, TransLoadMode, ListPlanes

    Transition = "Zirconia-cubic-mono-type2"
    ListPlanes = []
    TransLoadMode = 1
    LoadTransitionFiles()

#====================================================================================================

def NiTi():

    global Transition, TransLoadMode, ListPlanes

    Transition = "NiTi"
    ListPlanes = []
    TransLoadMode = 1
    LoadTransitionFiles()

#====================================================================================================

def G():

    global Transition, TransLoadMode, ListPlanes

    Transition = "Fe-G"
    ListPlanes = ListPlanesCubic
    TransLoadMode = 1
    LoadTransitionFiles()
    
#====================================================================================================

def Sigma3():

    global Transition, TransLoadMode, ListPlanes
    
    Transition = "Sigma3"
    ListPlanes = ListPlanesCubic
    TransLoadMode = 1
    LoadTransitionFilesSigma3()

#====================================================================================================

def LoadParameters():
    global Transition, TransLoadMode
    global AngTolGrain, AngTolParentNucl, AngTolParentGrowth, MinGrainSize
    global NbMinGrd, PercentRecLimit, AngTolParentNuclStep, AngTolParentGrowthStep
    global AngTolGBStraight,AngTolParentNuclLimit,AngTolParentGrowthLimit, FlagAutoSave
    global FlagSupPix, FlagAutoSave, RecMethod, DPhaseMethod, FlagPolish, FlagHoles, ImageSaveMode
    global FrameHigh

    try: 
        os.chdir (EBSDDirectory)
        fname = str("parameters.arpge")
        f = open(fname, 'r')
        info = pickle.load(f)
        f.close()
        Transition, AngTolGrain, AngTolParentNucl, AngTolParentGrowth, MinGrainSize = info[0]
        NbMinGrd, PercentRecLimit, AngTolParentNuclStep, AngTolParentGrowthStep, AngTolGBStraight = info [1]
        AngTolParentNuclLimit,AngTolParentGrowthLimit = info[2]
        flaga, flagsuppix, flagautosave, recmethod, dphasemethod, flagpolish, flagholes = info[3]
        
        EtatTrans.set(Transition)
        EtatAngTolGrain.set(AngTolGrain)
        EtatAngTolParentGrowth.set(AngTolParentGrowth)
        EtatAngTolParentNucl.set(AngTolParentNucl)
        EtatMinGrainSize.set(MinGrainSize)
        TransLoadMode = 1
        FlagAutoSave.set(flaga)
        FlagSupPix.set(flagsuppix)
        FlagAutoSave.set(flagautosave)
        RecMethod.set(recmethod)
        DPhaseMethod.set(dphasemethod)
        FlagPolish.set(flagpolish)
        FlagHoles.set(flagholes)
        LoadTransitionFiles()
        
    except:
        m = "The parameter file does not exist"
        tkMessageBox.showinfo("Warning", m)
        
#====================================================================================================

def SaveParameters():
      
    try:
        info = [None, None, None, None]
        info[0] = Transition, AngTolGrain, AngTolParentNucl, AngTolParentGrowth, MinGrainSize 
        info[1] = NbMinGrd, PercentRecLimit, AngTolParentNuclStep, AngTolParentGrowthStep, AngTolGBStraight
        info[2] = AngTolParentNuclLimit,AngTolParentGrowthLimit 
        info[3] = FlagAutoSave.get(), FlagSupPix.get(), FlagAutoSave.get(), RecMethod.get(), DPhaseMethod.get(), FlagPolish.get(), FlagHoles.get()
        os.chdir (EBSDDirectory)
        fname = str("parameters.arpge")
        f = open(fname, 'w')
        pickle.dump(info, f)
        f.close()
    except:
        m = "Problem to save the parameter file"
        tkMessageBox.showinfo("Warning", m)
    
#====================================================================================================

def LoadTransitionFiles(showlist=1):

    global listOpDisor, Transition, TransLoadMode
    global CrystP, CrystD
    global TSD, TMD, listOpTarget, listOpSource, listOpVar, TabCompoOp
    global TSP, TMP
    global invTSD, invTSP
    global GroupeRotD, QuotientDir, PtoD
    global GroupeRotP, QuotientDual
    global listMatDtoP, listQPos
    global FEulersP, FEulersD
    global NextStep, listOp180, ButtonTri, ButtonBain, ButtonCPP, ButtonCPD, ButtonBurst, listButtons1
    global IPFaxesD,IPFaxesP

    oname = ""
    listOp180 = []
    listOpDisor = []
    os.chdir (os.path.join(TransitionDirectory, "ListAllTransitions"))
    if not TransLoadMode:
        oname = askopenfilename(filetypes=[("text files", "*.trans")])
        Transition = oname[oname.rfind("/") + 1:]
        Transition = Transition.split('.trans')[0]
        print "Transition : ",Transition
    if not oname:
        oname = os.path.join(TransitionDirectory, "ListAllTransitions", Transition + ".trans")
    f = open(oname, 'r')
    info = pickle.load(f)
    name, fileMat, filePrec, p1Prec, p1Mat, p2Prec, p2Mat, d1Prec, d1Mat, d2Prec, d2Mat = info
    f.close()

    TransLoadMode = 0

    os.chdir (os.path.join(TransitionDirectory, str(Transition)))
    f = open('CC_listRotDir.txt', 'r')
    listOpDisor = pickle.load(f)
    f.close()

    f = open('CC_PhaseTransition.txt', 'r')
    PtoD, QuotientDir, QuotientDual, TabVar, listOpTarget, Tab = pickle.load(f)
    f.close()

    CrystP = importCryst(fileMat, PhaseDirectory)
    CrystD = importCryst(filePrec, PhaseDirectory)

    TMP, TSP, GP = CrystP.metricTensDir, CrystP.structTensDir, CrystP.sym()
    TMD, TSD, GD = CrystD.metricTensDir, CrystD.structTensDir, CrystD.sym()

    FEulersP = DictPGFEulers[CrystP.pg]
    FEulersD = DictPGFEulers[CrystD.pg]
      
    invTSD = inv(TSD) 
    invTSP = inv(TSP)
            
    IPFaxesD = IPFaxes(CrystD)
    IPFaxesP = IPFaxes(CrystP)

    listOpVar = []
    for el in TabVar:
        #print el[1:][0]
        listOpVar.append(el[1:][0]) #car le premier element est le codage de loperateur sous forme vt si op = v1vt
##    print
##    print "list of operators as list of targets: Op = {v1vt} = {vt}"
##    for el in listOpTarget: print el # codage vt des operateurs sous forme op = v1vt = 1ere colonne de TabVar
##    print

    #print "list of sources Op = {vsv1}"
    listOpSource = []
    for el in listOpVar:
        s = []
        for vij in el:
            if vij[1] == 1: s.append(vij[0])
        listOpSource.append(s)

    #ne garde que les matrices a determinant positif
    GroupeRotD = []
    for g in GD:
        if det(g) > 0:
            GroupeRotD.append(g)
    #ne garde que les matrices a determinant positif
    GroupeRotP = []
    for g in GP:
        if det(g) > 0:
            GroupeRotP.append(g)

    CalcTabCompoOp()

    DtoP = inv(PtoD)
    listMatDtoP = []
    for lmat in QuotientDual:
        for mat in lmat:
            if det(mat) > 0:
                listMatDtoP.append(dot(mat, DtoP)) # prend la 1ere matrice dans chaque coset du dual qui est une rotation et la multiplie par T
                break
            
    listQPos = []
    for lq in QuotientDir:
        for q in lq:
            if det(q) > 0:
                listQPos.append(q)
                break

    EtatTrans.set(Transition)
    try: EtatListOp.delete(0, END)
    except: pass
    NextStep += 1

    for i in range(len(listOpDisor)):
        angle, axe = listOpDisor[i][0][0], listOpDisor[i][0][1]
        #listOpDisor[i] = [[ang(min),axe,listmat],[ang,axe,listmat],etc.] 
        axeres = SimpleAxe(axe, GD)
        if angle:
            axeres = IntegriseAxe(array(axeres))
            text = str("Op. " + str(i) + " = " + str(angle)[:4] + " Deg /  [" + str(axeres[0]) + " " + str(axeres[1]) + " " + str(axeres[2]) + "]")
        else:
            text = str("Op. " + str(i) + " = " + str(angle)[:4] + " Deg " )
        if showlist: EtatListOp.insert(END, text)
        if angle == 180: listOp180.append(i)

        # ajout des matrices des matrices inverses pour les operateurs ambivalents 
        for j in range(1, len(listOpDisor[i])):
            angj = listOpDisor[i][j][0]
            if abs(angj - angle) < 0.00001:
                listOpDisor[i][0][2].extend(listOpDisor[i][j][2])
            else: break
            j += 1

    if showlist and (Transition =="Fe-NW" or Transition == "Fe-KS" or Transition == "Fe-GT" or Transition == "Fe-G" or Transition == "Fe-Pitsch"):
        ButtonBain = Button(frameMenuCV, text=" Bain Packets ", font=("Helvetica", 8), height=1, width = 8, foreground="black", background="pink")
        ButtonBain.bind("<Button>", ShowBainZones)
        ButtonBain.grid (row=0, column=4, columnspan=1, ipadx=5, ipady=5)
        listButtons1.append(ButtonBain)

        ButtonCPP = Button(frameMenuCV, text=" CPP Packets ", font=("Helvetica", 8),height=1, width = 8, foreground="black", background="pink")
        ButtonCPP.bind("<Button>", ShowCPPPackets)
        ButtonCPP.grid (row=0, column=5, columnspan=1, ipadx=5, ipady=5)
        listButtons1.append(ButtonCPP)
        
        ButtonCPD = Button(frameMenuCV, text=" CPD Packets ", font=("Helvetica", 8),height=1, width = 8, foreground="black", background="pink")
        ButtonCPD.bind("<Button>", ShowCPDPackets)
        ButtonCPD.grid (row=0, column=6, columnspan=1, ipadx=5, ipady=5)
        listButtons1.append(ButtonCPD)

        ButtonBurst = Button(frameMenuCV, text=" Burst Packets ", font=("Helvetica", 8),height=1, width = 8, foreground="black", background="pink")
        ButtonBurst.bind("<Button>", ShowBurstZone)
        ButtonBurst.grid (row=0, column=7, columnspan=1, ipadx=5, ipady=5)
        listButtons1.append(ButtonBurst)
        
        ButtonTri = Button(frameMenuCV, text="RGB+", height=1, width = 7, foreground="black", background="violet")
        ButtonTri.bind("<Button>", ShowMapTri)
        ButtonTri.grid (row=0, column=8, columnspan=1, ipadx=5, ipady=5)
        listButtons1.append(ButtonTri)

    if showlist and (Transition =="Zirconia-cubic-mono-type1" or Transition =="Zirconia-cubic-mono-type2"):
        ButtonTetra = Button(frameMenuCV, text="packets100", font=("Helvetica", 8), height=1, width = 8, foreground="black", background="pink")
        ButtonTetra.bind("<Button>", ShowPackets100)
        ButtonTetra.grid (row=0, column=4, columnspan=1, ipadx=5, ipady=5)
        listButtons1.append(ButtonTetra)
        ButtonTetra = Button(frameMenuCV, text="packets010", font=("Helvetica", 8), height=1, width = 8, foreground="black", background="pink")
        ButtonTetra.bind("<Button>", ShowPackets010)
        ButtonTetra.grid (row=0, column=5, columnspan=1, ipadx=5, ipady=5)
        listButtons1.append(ButtonTetra)
        ButtonTetra = Button(frameMenuCV, text="packets001", font=("Helvetica", 8), height=1, width = 8, foreground="black", background="pink")
        ButtonTetra.bind("<Button>", ShowPackets001)
        ButtonTetra.grid (row=0, column=6, columnspan=1, ipadx=5, ipady=5)
        listButtons1.append(ButtonTetra)
        
    frameMenuCV.update()
    BuildMenu()

#====================================================================================================

def LoadTransitionFilesSigma3():

    global Transition
    global TSD, TMD, TSP, TMP
    global invTSD, CrystD, CrystP
    global GroupeRotD, QuotientDir, PtoD, GroupeRotP, QuotientDual
    global listTwinCompo #table de composition des operateurs jusqua n<N/2
    global listOpDisor # # liste des matrices op [angmin,axe, lmat] jusqua N
    global listTwinNom # liste des noms des operateurs -nom- = meme taille que listOpDisor
    global listTwinNum # liste des numeros de ordres des operateurs
    global DictTwinNom #Dictionnaires donnant le nom de loperateur a partir dune chaine
    global DictTwinPos #Dictionnaires donnant le numero de loperateur a partir du nom
    global FEulersD, FEulersP
    global ListPlanes
    global AngTolGrain, AngTolParentGrowth
    global NextStep

    EtatRun.set("\n"+"Loading..."+"\n")
    LabelRun["fg"] = "red"
    frameFile.update()

    Transition = "Sigma3"
    AngTolGrain = 3.0
    EtatAngTolGrain.set(AngTolGrain)
    AngTolParentGrowth = 3.0
    EtatAngTolParentGrowth.set(AngTolParentGrowth)

    CrystD = importCryst("Cu", PhaseDirectory)

    TMD, TSD, GD = CrystD.metricTensDir, CrystD.structTensDir, CrystD.sym()
    TMS = Identity # pour simplifier les calculs, pas besoin de prendre les tenseurs du cuivre
    TSD = Identity
    invTSD = Identity

    GroupeRotD = []
    for g in GD:
        if det(g) > 0:
            GroupeRotD.append(g)

    TMP = TMD
    TSP = TSD
    CrystP = CrystD
    GroupeRotP = GroupeRotD

    ListPlanes = [[1, 1, 1], [1, 1, 0], [1, 1, 2]]

    os.chdir (os.path.join(TransitionDirectory, str(Transition)))

    f = open('CC_PhaseTransition.txt', 'r')
    PtoD, QuotientDir, QuotientDual, TabVar, listOpTarget, Tab = pickle.load(f) # a prevoir d importer T et la list des configs equivalentes
    f.close()

##    f = open('CC_listTwinRot.txt','r')
##    listTwinOpRot = pickle.load(f) #liste de tous les operateurs a la serie [angmin,axe, lmat] sans faire de classe par ordre de maclage
##    f.close()
    f = open('CC_listTwin_Class.txt', 'r')
    listTwinOp = pickle.load(f) #liste de tous les operateurs classe par ordre de maclage [ [[op,lch,ang,axe,nom]...] [...]   ]
    f.close()
    f = open('CC_listTwinTabCompo.txt', 'r')
    listTCompo = pickle.load(f)
    f.close()

    FEulersP = DictPGFEulers[CrystD.pg]
    FEulersD = DictPGFEulers[CrystD.pg]

    order = 2
    sousorder = 1
    nop = 3
    #print listTCompo[order][sousorder][nop] # est fait pour faciliter l affichage en tableau mais pas pratique cette decomposition => listTwinComp a reformater
    listTwinCompo = []
    for i in range(len(listTCompo)):
        for j in range(len(listTCompo[i])):
            lc = []
            for k in range(len(listTCompo[i][j])):
                for el in listTCompo[i][j][k]:
                    lc.append(el)
            listTwinCompo.append(lc)
    # REFAIT LA TABLE POUR QUELLE COMMENCE A SIGMA1
    listnom = []
    i = 0
    flag = 0
    for lop in listTwinOp:
        for op, lch, ang, axe, nom in lop:
            listnom.append([nom])
            if i >= len(listTwinCompo):
                flag = 1
                break
            i += 1
        if flag:break


    for i in range(len(listTwinCompo)): #rajoute la 1ere colonne
        listTwinCompo[i].insert(0, listnom[i])
    listTwinCompo.insert(0, listnom) # rajoute la 1ere ligne


    listOpVar = []
    for el in TabVar:
        listOpVar.append(el[1:][0]) #car le premier element est le codage de loperateur sous forme vt si op = v1vt

    #print "list of sources Op = {vsv1}"
    listOpSource = []
    for el in listOpVar:
        s = []
        for vij in el:
            if vij[1] == 1: s.append(vij[0])
        listOpSource.append(s)

    EtatTrans.set(Transition)
    EtatListOp.delete(0, END)

    listOpDisor = [] # liste des matrices op [angmin,axe, lmat]
    listTwinNom = [] # liste des noms des operateurs [nom,lch] = meme taille que listOpDisor
    listTwinNum = []
    n = 0
    icumul = 0
    for lop in listTwinOp:
            EtatListOp.insert(END, "=========== Sigma 3 exp" + str(n) + " ============")
            stepg = max(len(listTwinOp)/StProg,1)
            UpDateProgress((n * 100) / len(listTwinOp))
            i = 0
            for op, lch, ang, axe, nom in lop:
                if n == 0:
                    listOpDisor.append([[0, [1, 0, 0], GroupeRotD]])
                    listTwinNom.append(["0a", [[]]])
                elif n <= 6: # au dela ls calculs prenent trop de temps
                    m = dot(1. / 3 ** n, op) #attention op = matrice de codage, voir papier acta cryst
                    # m  = rotation mais pas encore la matrice de disorientation
                    dism = Disorientation(m, GroupeRotD)
                    listm = []
                    for s in GroupeRotD:
                        newm = dot(inv(s), dot(dism, s))
                        flag = 1
                        for mm in listm:
                            if allclose(mm, newm):
                                flag = 0
                                break
                        if flag: listm.append(newm)
                    listOpDisor.append([[ang, axe, listm]])
                    listTwinNom.append([nom, lch])
                    #print nom,lch[0]
                i += 1
                axeres = SimpleAxe(axe, GroupeRotD)
                text1 = str(n) + " " + nom[len(str(n)):] + " = " + "'" + stringify(lch[0]) + "'"
                text2 = str("     = Rotation " + str(round(ang, 2)) + " Deg. / " + str(axeres))
                EtatListOp.insert(END, text1)
                EtatListOp.insert(END, text2)
            n = n + 1
            listTwinNum.append(icumul)
            icumul = icumul + i
    # "-------------------------------------------------------------"
    DictTwinNom = {} #permet de retrouver le nom de loperateur a partir de la chaine
    DictTwinPos = {} #permet de retrouver le numerom de loperateur a partir de la chaine
    n = 0
    for lop in listTwinOp:
            for op, lch, ang, axe, nom in lop:
                for ch in lch:
                    s = stringify(ch)
                    DictTwinNom[s] = nom
                DictTwinPos[nom] = n
                n = n + 1
    # "-------------------------------------------------------------"
    EtatRun.set("\n"+"Open the EBSD file"+"\n")
    LabelRun["fg"] = "blue"
    UpDateProgress(0)
    frameFile.update()
    NextStep += 1
    BuildMenu()

#===================================================================================================

def ReLoadHKL():
    global ReLoadFlag, phD, phP, phMax1, phMax2

    if DPhaseMethod.get()==0:
        DPhaseMethod.set(1)
        phD = phMax2
        phP = phMax1
    elif DPhaseMethod.get()==1:
        DPhaseMethod.set(0)
        phD = phMax1
        phP = phMax2
    FlagSupPix.set(0)
    FlagHoles.set(0)
    FlagPolish.set(0)
    ReLoadFlag=1
    openFileHKL()

#===================================================================================================
##
##def KeepOnlyDaughter():
##    global ReLoadFlag, phD, phP, phMax1, phMax2
##
##    phP = 0
##    FlagSupPix.set(0)
##    FlagHoles.set(0)
##    FlagPolish.set(0)
##    ReLoadFlag=1
##    openFileHKL()

#===================================================================================================

def openFileHKL():
    
    global Longueur, Nbpixx, Nbpixy, XStep, YStep, TabPix, TabPixP, TimeStart
    global euler1min, euler2min, euler3min, deuler1, deuler2, deuler3, bcmin, dbc, FileName, FullFileName, Scale
    global euler1minP, euler2minP, euler3minP, deuler1P, deuler2P, deuler3P
    global TSD,TSP, invTSD, invTSP 
    global Xmax, Ymax
    global SourceFile, OptShowB
    global NextStep
    global Entete,iPhase, phD, phP, phMax1, phMax2

    OptShowB = "pixels"
    TabPix =[]
    
    if not ReLoadFlag:
        info = "Do not forget: choose a phase transition before reconstructing the daughter grains"
        if Transition == "?":
            tkMessageBox.showinfo("Help", info)
            return
        EtatRun.set("\n"+"Opening the file..."+"\n")
        LabelRun["fg"] = "red"
        frameFile.update()

        TimeStart = time.clock()
        try: os.chdir (os.path.join(EBSDDirectory, "maps_CTF"))
        except: os.chdir (os.path.join(EBSDDirectory, "maps_HKL"))
        oname = askopenfilename(filetypes=[("ctf files", "*.ctf")])
        if oname :
            SourceFile = 'HKL'
            EtatRun.set("\n"+"Opening the file. Please wait..."+"\n")
            # will be used for the Entete
            if Transition == "Fe-NW" or Transition == "Fe-GT" or Transition == "Fe-G" or Transition == "Fe-KS" :
                phasetext = HKL_FeGamma
            elif Transition == "Burgers":
                phasetext = HKL_TiBeta
            elif Transition == "Zirconia-cubic-mono-type1" or Transition == "Zirconia-cubic-mono-type2":
                phasetext = HKL_ZrO2_cubic
            else:
                phasetext = "Phases	1\n"+cut(CrystD.a,5)+";"+cut(CrystD.b,5)+";"+cut(CrystD.c,5)+"	"+cut(DEGREE*CrystD.alpha,5)+";"+cut(DEGREE*CrystD.beta,5)+";"+cut(DEGREE*CrystD.gamma,5)+"	"+str(CrystD.el)+"     \n"
                phasetext = phasetext + t_HKL
            #====================
            
            frameFile.update()
            FileName = oname
            FullFileName = oname
            bufile = open(FileName)
            FileName = str(FileName[:len(FileName) - 4])
            FileName = FileName[FileName.rfind("/") + 1:]
            #print FileName
            lines = bufile.readlines()
            bufile.close()
            lenTot = len(lines)
            #print "nb lines in the file",lenTot
            nb_col = arange(11)
            i = 0
            iPhase = 0
            Entete = ""
            stopEnt = 0
            #cherche le depart dans le 50 premieres lignes
            for i in range(50) :
                a = string.split(lines[i])
                if "Author" in a:
                    text = "Author [from ARPGE]\n"
                elif "Phases" in a:
                    text = phasetext
                    Entete = Entete + text 
                    stopEnt =1
                else: text = lines[i]
                if not stopEnt: Entete = Entete + text 
                
                if "XCells" in a:
                    nx = string.atoi(a[1])
                elif "YCells" in a:
                    ny = string.atoi(a[1])
                elif "XStep" in a:
                    xstep = a[1]
                    xstep = xstep.replace(",", ".")
                    xstep = string.atof(xstep)
                elif "YStep" in a:
                    ystep = a[1]
                    ystep = ystep.replace(",", ".")
                    ystep = string.atof(ystep)
                elif "Phase" in a:
                    iPhase = i
                    break    
                i = i + 1
                x = str(oname)   
            Longueur = lenTot - iPhase - 1
            Nbpixx = nx
            Nbpixy = ny
            XStep = xstep
            YStep = ystep
        else:
            EtatRun.set("\n"+"Cannot open the file!"+"\n")
            LabelRun["fg"] = "red"
            frameFile.update()        

        print "Header of the .ctf file"
        print Entete
        print
        print "Nb of pixels", Longueur
        print "Nbpixx ,Nbpixy = ", Nbpixx, Nbpixy
        print "iPhase ", iPhase
        
        # les donnees angulaire viennent apres la ligne iPhase
        # tout ce qui est avant est l Entete sur fichier
        #Entete = lines[:i]
        i = iPhase + 1
        ii = 0

        #1er passage pour voir quelle est la phase majoritaire = phase fille
        listph = []
        listphT = []
        stepg = lenTot/StProg
        while (i < lenTot) :
            if not i%stepg: UpDateProgress((i * 100) / lenTot)
            a = string.split(lines[i])
            for j in range(11):
                if "," in a[j]:
                    c = string.split(a[j], ',')
                    a[j] = "%1d.%1d" % (string.atoi(c[0]), string.atoi(c[1]))
            ph = string.atoi(a[0])
            if ph: # les points indexes
                if ph not in listph:
                    listph.append(ph)
                    listphT.append(1)
                else:
                    index = listph.index(ph)
                    listphT[index] += 1
            i += 1

        maxi = max(listphT)
        index = listphT.index(maxi)
        phMax1 = listph[index]

        try:
            listphT2 = copy.deepcopy(listphT)
            listphT2.remove(maxi)
            maxi2 = max(listphT2)
            index2 = listphT.index(maxi2)
            phMax2 = listph[index2]
        except:
            phMax2=0
            
            
        if DPhaseMethod.get()==0:
            print "the daughter phase is supposed to be majoritary" + ", it is " + str(phMax1)
            phD = phMax1
            phP = phMax2
            
        if DPhaseMethod.get()==1:
            try:
                print "the daughter phase is supposed to be minority" + ", it is " + str(phMax2)
                phD = phMax2
                phP = phMax1
            except:
                print "error in the attribution of the phase number"
            
        UpDateProgress(0)

    #----------ReLoad or not------------------ 
    #2eme passage pour ne garder que les donnees de la phase fille
    print "the daughter phase chosen for the reconstruction is ", phD
    if phP!=0: print "the retained phase chosen for the reconstruction is ", phP

    TabPix = []
    TabPixP = []
    
    euler1min,euler1max,euler2min,euler2max,euler3min, euler3max = 360,0,360,0,360,0
    euler1minP,euler1maxP,euler2minP,euler2maxP,euler3minP, euler3maxP = 360,0,360,0,360,0

    bcmin = 255
    bcmax = 0
    Xmax = 0
    Ymax = 0

    percD, percP = 0,0 

    if ReLoadFlag:
            bufile = open(FullFileName)
            lines = bufile.readlines()
            bufile.close()
    
    i = iPhase + 1
    
    try:
        stepg = lenTot/StProg
    except:
        lenTot = Longueur+iPhase+1
        stepg = lenTot/StProg

    while (i < lenTot) :
        if not i%stepg: UpDateProgress((i*100)/lenTot)
        a = string.split(lines[i])
        for j in range(11):
            if "," in a[j]:
                c = string.split(a[j], ',')
                a[j] = "%1d.%1d" % (string.atoi(c[0]), string.atoi(c[1]))
        try: el = map(float, a) #temp
        except: pass
        # Channel 5 ctf
        #Phase	X	Y	Bands	Error	Euler1	Euler2	Euler3	MAD	BC	BS
        #pixel = Pixel(el[5],el[6],el[7],ii,0,0) # prend trop de place memoire si on code sous cette forme
        ph = string.atoi(a[0])
        if ph == phD:
            pix = [el[5], el[6], el[7], 0, 0, int(el[9])] #e1,e2,e3,0,0, bc] 0,0 pour futurs graind et grainp
            if el[5] < euler1min : euler1min = el[5]
            elif el[5] > euler1max : euler1max = el[5]
            if el[6] < euler2min : euler2min = el[6]
            elif el[6] > euler2max : euler2max = el[6]
            if el[7] < euler3min : euler3min = el[7]
            elif el[7] > euler3max : euler3max = el[7]
            if el[9] < bcmin : bcmin = el[9]
            elif el[9] > bcmax : bcmax = el[9]
            percD+=1
        else:
            if ph == phP and phP!=0: pix = [0, 0, 0, 0, 0, 300] # 300 = retained parent
            else : pix = [0, 0, 0, 0, 0, int(el[9])]

        TabPix.append(pix)
        
        if ph == phP and phP!=0:
            pixP = [el[5], el[6], el[7], 0, 0, int(el[9])] #e1,e2,e3,0,0, bc] 0,0 pour futurs graind et grainp
            if el[5] < euler1minP : euler1minP = el[5]
            elif el[5] > euler1maxP : euler1maxP = el[5]
            if el[6] < euler2minP : euler2minP = el[6]
            elif el[6] > euler2maxP : euler2maxP = el[6]
            if el[7] < euler3minP : euler3minP = el[7]
            elif el[7] > euler3maxP : euler3maxP = el[7]
            percP+=1
##            if el[9] < bcmin : bcmin = el[9]
##            elif el[9] > bcmax : bcmax = el[9]          
        else:
            pixP = [0, 0, 0, 0, 0, int(el[9])] #daughter pixel
        TabPixP.append(pixP)

##        if el[1] > Xmax : Xmax = el[1]
##        if el[2] > Ymax : Ymax = el[2]  #10 mai 2020

        if (((i) % 50000) == 0):
            print "i = ", (i), "/", (lenTot)
        i+=1

    deuler1 = euler1max - euler1min
    deuler2 = euler2max - euler2min
    deuler3 = euler3max - euler3min
    dbc     = bcmax     - bcmin

    deuler1P = euler1maxP - euler1minP
    deuler2P = euler2maxP - euler2minP
    deuler3P = euler3maxP - euler3minP

    EtatFile.set(FileName)
    EtatSize.set(str(Nbpixx) + "x " + str(Nbpixy) + " pixels")

    Longueur = len(TabPix)
    print "Nb of pixels", Longueur

    Xmax = Nbpixx * XStep
    Ymax = Nbpixy * YStep # 10 mai 2020

    #janv2020
    print "SourceFile : ",SourceFile
    if (SourceFile=="HKL" or SourceFile=="ARPGE_HKL") and CrystD.syst == "Hexagonal"  :
        Pas = array([[0,-1,0],[1,0,0],[0,0,1]])
        TSD = dot(Pas,TSD)
        invTSD = inv(TSD) 
    
    if (SourceFile=="HKL" or SourceFile=="ARPGE_HKL") and CrystP.syst == "Hexagonal"  :
        Pas = array([[0,-1,0],[1,0,0],[0,0,1]])
        TSP = dot(Pas,TSP)
        invTSP = inv(TSP)

    lines = [] #remet a zero la memoire a verifier
    os.chdir(EBSDDirectory)

    if not HideFlag.get(): showmap()
    UpDateProgress(0)

    percD = percD*100/Longueur
    percP = percP*100/Longueur

    if percP >0 :
        txt1= "There is more than one phase in the map.\n"
        text2 = "The major phase was supposed to be the daughter phase; if it is not so, choose Reload ctf by changing Major <-> Minor \n"
        #text3 = "If you want to keep only the daughter phase and ignore all the other phases (for dilatation), choose Reload and keep only Daughter Phase \n"
        tkMessageBox.showinfo("Info", txt1+text2)

    EtatRun.set("\n"+"Daughter = "+ str(percD)+ "%, Retained Parent = "+ str(percP)+"%."+" Next step: Suppress spikes and/or unindexed pixels"+"\n")
    LabelRun["fg"] = "blue"
    NextStep += 1



#====================================================================================================  

def ReadEntete(newrep,FName,subset):
    print "ReadEntete used"
    os.chdir (newrep)              
    bufile = open(FName + ".ctf")
    i = 0
    Entete = ""
    if not subset:
        for line in bufile.readlines(500) :
            a = string.split(line)
            if "Author" in a:
                text = "Author [from ARPGE]\n"
            else : text = line
            Entete = Entete + text
            if "Phase" in a:
                break
            i = i + 1
            if i > 400: print "ERROR in reading the header"
    else:
        for line in bufile.readlines(500) :
            a = string.split(line)
            if "Author" in a:
                text = "Author [from ARPGE]\n"
            elif "XCells" in a:
                text = "XCells\t" + str(Nbpixx) + '\n'
            elif "YCells" in a:
                text = "YCells\t" + str(Nbpixy) + '\n'
            else : text = line
            Entete = Entete + text
            if "Phase" in a:
                break
            i = i + 1
            if i > 400: print "ERROR in reading the header"
    bufile.close()
    return Entete

#====================================================================================================  

def exportParentGrains ():
    print "SourceFile to export the map :", SourceFile
    if SourceFile == 'HKL':exportParentToHKL()
    elif SourceFile == 'TSL':exportParentToTSL()
    elif SourceFile =="ARPGE_TSL": exportParentToTSL()
    elif SourceFile =="ARPGE_HKL": exportParentToHKL()
    else: print "Problem to export the file : please contact C. Cayron"

#====================================================================================================  

def exportParentPixels ():
    if SourceFile == 'HKL':exportParentPixToHKL()
    elif SourceFile == 'TSL':exportParentPixToTSL()
    elif SourceFile =="ARPGE_TSL": exportParentPixToTSL()
    elif SourceFile =="ARPGE_HKL": exportParentPixToHKL()
    else: print "Problem to export the file : please contact C. Cayron"
    
#====================================================================================================  

def exportParentToHKL (optexport=0):
    global Entete


    if SourceFile == 'TSL':
        tkMessageBox.showinfo("Not possible:", 'the file of daughter grains was acquired with TSL')
        return
    if SourceFile == 'ARPGE' or SourceFile == 'ARPGE_TSL' or SourceFile == 'ARPGE_HKL':
        print "The loaded file is a .arpge file"

    try:
        EtatRun.set("\n"+"Creating the file..."+"\n")
        LabelRun["fg"] = "red"
        frameFile.update()

        subset = 0
        FName = FileName
        if FName[-2]== "v" and FName[-3]== "_": #version incrementee, numero a retirer pour trouver le ctf original
            FName = FName[:-3]
        print "***",FileName, EBSDDirectory
        #print "cherche", FName+".ctf", "dans", EBSDDirectory+"/maps_CTF"
        newrep = findfile(FName, os.path.join(EBSDDirectory, "maps_CTF"))
        if newrep=="?": newrep = findfile(FName, os.path.join(EBSDDirectory, "maps_HKL"))
        try:
            if 'subset' in FName:
                subset=1
                print "subset identified"
            FName = FName.split('subset')[0]
            newrep = findfile(FName, os.path.join(EBSDDirectory, "maps_CTF"))
            if newrep=="?": newrep = findfile(FName, os.path.join(EBSDDirectory, "maps_HKL"))
        except: pass
        print "file :", FName, " \n localized in repertory :",newrep

        if len(Entete)==0: # case of a file open from an .arpge file
            print "No header, new header created"
            Entete = ReadEntete(newrep,FName,subset)
     
        z='00000'
        def compl(f,n=6):
            # complete f=1.2 en 1.2000 pour avoir 6 caracteres
            if f: s=str(f)[:n]
            else: s = "0.0000"
            return s[:n]+z[:n-len(s)]

        def chgEnteteBS(word):
            global Entete
            print "header modified in the BS column"
            NewEntete = ""
            Lines = Entete.split("\n")
            for line in Lines:
                if "Phase" in line and "MAD" in line and "BC" in line:
                    nline = line.replace("BS",word)
                else: nline = line + "\n"
                NewEntete = NewEntete + nline 
            Entete = NewEntete
    
        p = EBSDDirectory + '/Results'
        os.chdir (p)
        if not optexport: ffname = FileName + "_ParentGrains.ctf"
        elif optexport==1:
            ffname = FileName + "_Bain_Packets.ctf"
            chgEnteteBS("Bain")
        elif optexport==2:
            ffname = FileName + "_CPP_Packets.ctf"
            chgEnteteBS("CPP")
        elif optexport==3:
            ffname = FileName + "_CPD_Packets.ctf"
            chgEnteteBS("CPD")
        ff = open(ffname, 'w')
        ff.write(Entete)
        X, Y = 0, 0
        Band = 6
        Error = 0
        MAD = compl(0.1)
        BC = 100

        for nb in range(len(TabPix)):
            BS = 200
            flagret = 0
            X, Y = PosXY(nb)
            X,Y = compl(X),compl(Y)
            if TabPix[nb][4]== 0:
                E1,E2,E3 = 0,0,0
                np = 0
            else:
                if TabPix[nb][4]>0: grp = TabParentGrains[TabPix[nb][4] - 1]
                else:
                    grp = TabRetainedGrains[abs(TabPix[nb][4]) - 1]
                    flagret = 1
                E1 = grp.euler1
                E2 = grp.euler2
                E3 = grp.euler3
            C2 = 1-float(TabGrains[TabPix[nb][3] - 1].confind)/30
            if C2<0: C2=0
            C2 = compl(C2)
            if E1==0 or TabGrains[TabPix[nb][3] - 1].confind==30: np = 0
            else: np = 1
            
            #===========================option to export the packets in place of the BS column ===============
            if optexport==1 and not flagret: # Bain packets
                try:
                    grd = TabGrains[TabPix[nb][3] - 1] #3 = graind
                    v = grd.variant[0]
                    if v:
                        if v == 1 or v == 4 or v == 6 or v == 12 or v == 17 or v == 19 or v == 21 or v == 23 : BS = 85 #color = "blue" in draw mode
                        elif v == 5 or v == 8 or v == 11 or v == 13 or v == 14 or v == 15 or v == 16 or v == 22 : BS = 170 #color = "green" in draw mode
                        elif v == 2 or v == 3 or v == 7 or v == 9 or v == 10 or v == 18 or v == 20 or v == 24 : BS = 255 #color = "red" in draw mode
                except: BS = 0 # packet not found
            elif optexport==2 and not flagret: # Morito CPP packets
                try:
                    grd = TabGrains[TabPix[nb][3] - 1] #3 = graind
                    v = grd.variant[0]
                    if v:
                        if v == 1 or v == 2 or v == 3 or v == 5 or v == 6 or v == 8 : BS = 64 #color = "blue"
                        elif v == 4 or v == 7 or v == 9 or v == 11 or v == 12 or v == 14: BS = 128 #color = "green"
                        elif v == 10 or v == 13 or v == 15 or v == 19 or v == 20 or v == 23: BS = 192 #color = "red"
                        elif v == 16 or v == 17 or v == 18 or v == 21 or v == 22 or v == 24: BS = 255 #color = "gold"
                except: BS = 0
            elif optexport==3 and not flagret: # CPD packets
                try:
                    grd = TabGrains[TabPix[nb][3] - 1] #3 = graind
                    v = grd.variant[0]
                    if v:
                        if v == 1 or v == 3 or v == 4 or v == 9: BS = 42 #color = "blue"
                        elif v == 5 or v == 6 or v == 16 or v == 17: BS = 84 #color = "green"
                        elif v == 11 or v == 12 or v == 13 or v== 23: BS = 126 #color = "red"
                        elif v == 7 or v == 14 or v == 18 or v== 22: BS = 168 #color = "gold"
                        elif v == 19 or v == 20 or v == 21 or v== 24: BS = 210 #color = "pink"
                        elif v == 2 or v == 8 or v == 10 or v== 15: BS = 255 #color = "chocolate"
                except: BS = 0
            #===================================================================================================
                
            #s = '%1d\t%3.4f\t%3.4f\t%1d\t%1d\t%3.3f\t%3.3f\t%3.3f\t%1.4f\t%3d\t%1d\n' % (1, X, Y, Band, Error, E1, E2, E3, MAD, BC, BS)
            s = '%1d\t%.6s\t%.6s\t%1d\t%1d\t%.6s\t%.6s\t%.6s\t%.6s\t%3d\t%1d\n' % (np, str(X), str(Y), Band, Error, compl(E1), compl(E2), compl(E3), str(MAD), BC, BS)
            ff.write(s)
        ff.close()
        EtatRun.set("\n"+"File " + ffname + " created in " + p +"\n")
        LabelRun["fg"] = "blue"
        os.startfile(p)
        os.chdir (EBSDDirectory)
        
    except:
        if optexport ==0: tkMessageBox.showinfo("ERROR:", 'Check there is no special letters in names of the folder tree and that the initial ctf file is located in the HKL folder. \n If the problem remains, contact C. Cayron')
        else: tkMessageBox.showinfo("ERROR:", 'Check that the packets have been calculated and shown. \n If the problem remains, contact C. Cayron')
#====================================================================================================  

def exportParentPixToHKL ():
    global Entete

    if SourceFile == 'TSL':
        tkMessageBox.showinfo("Not possible:", 'the file of daughter grains was acquired with TSL')
        return
    if not TabPixP:
        tkMessageBox.showinfo("Not possible:", 'the pixel map of the parent grains must be calculated')
        return
    try:   
        EtatRun.set("\n"+"Creating the file..."+"\n")
        LabelRun["fg"] = "red"
        frameFile.update()

        subset = 0
        FName = FileName
        if FName[-2]== "v" and FName[-3]== "_": #version incrementee, numero a retirer pour trouver le ctf original
            FName = FName[:-3]
        print "**",FileName, EBSDDirectory
        #print "cherche", FName+".ctf", "dans", EBSDDirectory+"/maps_CTF"
        newrep = findfile(FName, os.path.join(EBSDDirectory, "maps_CTF"))
        if newrep=="?": newrep = findfile(FName, os.path.join(EBSDDirectory, "maps_HKL"))
        try:
            FName = FName.split('subset')[0]
            newrep = findfile(FName, os.path.join(EBSDDirectory, "maps_CTF"))
            if newrep=="?": newrep = findfile(FName, os.path.join(EBSDDirectory, "maps_HKL"))
            subset=1
            print "subset identified"
        except: pass
        print "newfile=", FName, "new repertory",newrep

        #os.chdir (newrep)
        p = EBSDDirectory + '\Results'  #new way:: save the file in the Result folder
        os.chdir (p)
        bufile = open(FName + ".ctf")

        if len(Entete)==0: # case of a file open from an .arpge file
            print "Read the initial file to create the new one"
            Entete = ReadEntete(newrep,FName,subset)
            
        z='000000'

        def compl(f):
            # complete f=1.2 en 1.2000 pour avoir 6 caracteres
            s=str(f)[:6]
            return s[:6]+z[:6-len(s)]
            
        try: os.chdir (EBSDDirectory + '/maps_CTF')
        except: os.chdir (EBSDDirectory + '/maps_HKL')
        fname = FileName + "_ParentPixels.ctf"
        f = open(fname, 'w')
        f.write(Entete)
        X, Y = 0, 0
        Band = 6
        Error = 0
        MAD = compl(0.1)
        BC = 100
        BS = 200
        for nb in range(len(TabPixP)):
            X, Y = PosXY(nb)
            X,Y = compl(X),compl(Y)
            E1 = TabPixP[nb].euler1
            E2 = TabPixP[nb].euler2
            E3 = TabPixP[nb].euler3
            C2 = 1-float(TabGrains[TabPix[nb][3] - 1].confind)/30
            if C2<0: C2=0
            C2 = compl(C2)
            if E1==0: np = 0
            else: np = 1
            #s = '%1d\t%3.4f\t%3.4f\t%1d\t%1d\t%3.3f\t%3.3f\t%3.3f\t%1.4f\t%3d\t%1d\n' % (1, X, Y, Band, Error, E1, E2, E3, MAD, BC, BS)
            s = '%1d\t%.6s\t%.6s\t%1d\t%1d\t%.6s\t%.6s\t%.6s\t%.6s\t%3d\t%1d\n' % (np, str(X), str(Y), Band, Error, compl(E1), compl(E2), compl(E3), str(MAD), BC, BS)
            f.write(s)
        f.close()
        EtatRun.set("\n"+"File " + fname + " created in " + os.path.join(EBSDDirectory, "Results")+"\n")
        LabelRun["fg"] = "blue"
        try: os.startfile(EBSDDirectory + "/maps_CTF")
        except: os.startfile(EBSDDirectory + "/maps_HKL")
        os.chdir (EBSDDirectory)
    except:
        tkMessageBox.showinfo("ERROR:", 'Check there is no special letters in names of the folder tree and that the initial ctf file is located in the HKL folder. \n If the problem remains, contact C. Cayron')

#===================================================================================================

def openFileTSL():
    
    global Longueur, Nbpixx, Nbpixy, XStep, YStep, TabPix, TimeStart
    global euler1min, euler2min, euler3min, deuler1, deuler2, deuler3, bcmin, dbc, FileName, Scale
    global Xmax, Ymax, Units, TabPixP
    global GRID
    global SourceFile
    global NextStep

    info = "Dont forget: choose a phase transition before reconstructing the daughter grains"
    if Transition == "?":
        tkMessageBox.showinfo("Help", info)
        return
    EtatRun.set("\n"+"Opening the file..."+"\n")
    LabelRun["fg"] = "red"
    frameFile.update()

    TimeStart = time.clock()
    try: os.chdir (os.path.join(EBSDDirectory, "maps_ANG")) #after Oct.2019
    except: os.chdir (os.path.join(EBSDDirectory, "maps_TSL")) #before Oct.2019
    oname = askopenfilename(filetypes=[("ang files", "*.ang")])
    if oname :
        SourceFile = 'TSL'
        EtatRun.set("\n"+"Opening the file. Please wait..."+"\n")
        frameFile.update()
        FileName = oname
        bufile = open(FileName)
        FileName = str(FileName[:len(FileName) - 4])
        FileName = FileName[FileName.rfind("/") + 1:]
        #print "FileName=",FileName
        lines = bufile.readlines()
        bufile.close()
        lenTot = len(lines)
        #print "nb lines in the file",lenTot
        nb_col = arange(11)
        i = 0
        istart = 0
        ny,nxeven,nxodd = 0,0,0
        i = 0
        while 1:
            a = string.split(lines[i])
            if "GRID:" in a:
                if "HexGrid" in a:
                    GRID = "Hexagon"
                    FileName = FileName + "_HexGrid"
            elif "NROWS:" in a:
                ny = string.atoi(a[2])
            elif "NCOLS_EVEN:" in a:
                nxeven = string.atoi(a[2])
            elif "NCOLS_ODD:" in a:
                nxodd = string.atoi(a[2])
            elif "NCOLS:" in a:
                nxodd = string.atoi(a[2])
                nxeven = nxodd
            elif "XSTEP:" in a:
                xstep = a[2]
                xstep = xstep.replace(",", ".")
                XStep = string.atof(xstep)
            elif "YSTEP:" in a:
                ystep = a[2]
                ystep = ystep.replace(",", ".")
                YStep = string.atof(ystep)
            elif "SCANID:" in a:
                istart = i
            istart = i
            i+=1
            x = str(oname)
            if "#" not in a: break
        
        nx = min(nxeven, nxodd)

        print "ici je vois nx et ny = ", nx, ny

        #----------- Pour traiter les fichiers Edgar Rauch ASTAR-----------------
        if istart == 0: # Cas Edgar Rauch
            print "ASTAR file identified, the units should be in nm. Or it is .ang file in Sqr Grid without information..."
            Units = "nm"
            i=0
            while 1:
                a = string.split(lines[i])
                if "#" not in a:
                    istart = i-1 # -1 car ensuite on fera le +2 du cas general
                    break
                i+=1  
            a1 = string.split(lines[i+1])
            el = map(float, a1)
            XStep = el[3]
            nxp=0
            while 1:
                a = string.split(lines[i])
                el = map(float, a) 
                x,y = el[3],el[4]
                if y!=0:
                    nx = nxp
                    YStep = y
                    break
                i+=1
                nxp+=1
            af = string.split(lines[lenTot-1])
            el = map(float, af)
            ym = el[4]
            ny = int(ym/YStep+1)
        #-----------------------------------------------------------------------

        if nx ==0:
            print "new OIM version detected"
            oim = 1 # new version OIM
        else: oim =0 # old version OIM

        euler1min = 2 * math.pi
        euler1max = 0
        euler2min = 2 * math.pi
        euler2max = 0
        euler3min = 2 * math.pi
        euler3max = 0
        bcmin = 255
        bcmax = 0
        Xmax = 0
        Ymax = 0

        # les donnees angulaire viennent apres la ligne istart
        # tout ce qui est avant est l Entete sur fichier
        # Entete = lines[:i]

        #1er passage pour voir quelle est la phase majoritaire = phase fille
        i = istart 
        stepg =  lenTot/StProg
        listph = []
        listphT = []
        a1 = string.split(lines[istart])
        a2 = string.split(lines[istart+1])
        x1 = a1[3]
        x2 = a2[3]
        y1 = a1[4]

        if oim==1: #new oim version
            XStep = string.atof(x2)-string.atof(x1)
        
        while (i < lenTot) :
            if not i%stepg:UpDateProgress((i*100)/lenTot)
            a = string.split(lines[i])
            for j in range(len(a)): # len(a) = 10 sauf pour les fichiers Edgar Rauch = 9
                if "," in a[j]:
                    c = string.split(a[j], ',')
                    a[j] = "%1d.%1d" % (string.atoi(c[0]), string.atoi(c[1]))
            if oim:    
                if a[4] == y1: nx+= 1
                if a[3] == x1 or a[3] == x2: ny+= 1
            ph = a[7]
            if ph not in listph:
                listph.append(ph)
                listphT.append(1)
            else:
                index = listph.index(ph)
                listphT[index] = listphT[index] + 1
            i+=1

        if oim: nx = nx-1 # parce que jai compte les pixels de la 1ere lignes qui en ont plus que la 2d lignes

        Nbpixx = nx
        Nbpixy = ny

        pop =1
        # Modif here
        if nx%2 !=0 :
            pop = 0 #astuce pour fichier TSL

        maxi = max(listphT)
        index = listphT.index(maxi)
        phD = listph[index]

        print "the daughter phase is: ", phD
        print "the map size is: ", Nbpixx,"x",Nbpixy
        print "the step size is: ", XStep

        UpDateProgress(0)
       
        #2eme passage pour ne garder que les donnees de la phase fille
        i = istart
        ii = 1
        kl=0
        stepg = lenTot/StProg

        while (i < lenTot) :
            if not i%stepg: UpDateProgress((i*100)/lenTot)
            a = string.split(lines[i])
            for j in range(len(a)):
                if "," in a[j]:
                    c = string.split(a[j], ',')
                    a[j] = "%1d.%1d" % (string.atoi(c[0]), string.atoi(c[1]))
            el = map(float, a) #temp
            # ligne fichier .ang
            # 0.39122   1.82022   0.17254      0.00000      0.00000   67.4  0.917  0      1  0.712
            #pixel = Pixel(el[0],el[1],el[2],ii,0,0) # prend trop de place memoire si on code sous cette forme
            if a[7] == phD and el[6] > 0:
                pix = [el[0], el[1], el[2], 0, 0, el[5]]
                if el[3] > Xmax : Xmax = el[3]
                if el[4] > Ymax : Ymax = el[4]
                if el[0] < euler1min : euler1min = el[0]
                elif el[0] > euler1max : euler1max = el[0]
                if el[1] < euler2min : euler2min = el[1]
                elif el[1] > euler2max : euler2max = el[1]
                if el[2] < euler3min : euler3min = el[2]
                elif el[2] > euler3max : euler3max = el[2]
                if el[5] < bcmin : bcmin = el[5]
                elif el[5] > bcmax : bcmax = el[5]
            else:
                pix = [0, 0, 0, 0, 0, 0]   
                      
            TabPix.append(pix)
            TabPixP.append([0, 0, 0, 0, 0, 0]) # not finished, I must write it as it was done for HKL files
            
            if GRID == "Hexagon" and pop:
                if ii % (Nbpixx+1) == 0:
                    if kl%2==0:
                        TabPix.pop(-1) # pour avoir le meme nb de pixel par lignes
                        ii=0
                    else: ii=1
                    kl+=1

            if GRID == "Hexagon" and not pop:
                if ii % (Nbpixx+1) == 0:
                    if kl%2==1:
                        TabPix.pop(-1) # pour avoir le meme nb de pixel par lignes
                        ii=0
                    else: ii=1
                    kl+=1

            if (((i) % 50000) == 0):
                print "i = ", (i), "/", (lenTot)
            i+=1
            ii+=1

        Longueur = len(TabPix)

        deuler1 = (euler1max - euler1min) * DEGREE
        deuler2 = (euler2max - euler2min) * DEGREE
        deuler3 = (euler3max - euler3min) * DEGREE
        dbc = bcmax - bcmin

        EtatFile.set(FileName)
        EtatSize.set(str(Nbpixx) + " x " + str(Nbpixy) + " pixels")

        for pix in TabPix:
            pix[0] = pix[0] * DEGREE
            pix[1] = pix[1] * DEGREE
            pix[2] = pix[2] * DEGREE

        lines = [] #remet a zero la memoire a verifier
        os.chdir(EBSDDirectory)

        if not HideFlag.get(): showmap()
        UpDateProgress(0)

        EtatRun.set("\n"+"Next step: suppress spikes and/or unindexed pixels"+"\n")
        LabelRun["fg"] = "blue"
        NextStep += 1

    else:
        EtatRun.set("\n"+"Cannot open the file!"+"\n")
        LabelRun["fg"] = "red"
        frameFile.update()

#====================================================================================================  

def exportParentToTSL():
    
    if SourceFile == 'HKL':
        tkMessageBox.showinfo("Not possible:", 'the file of daughter grains was acquired with HKL')
        return
    if SourceFile == 'ARPGE':
        print "The loaded file is a .arpge file"
    try:
        EtatRun.set("\n"+"Creating the file..."+"\n")
        LabelRun["fg"] = "red"
        frameFile.update()

        subset = 0
        FName = FileName
        if FName[-2]== "v" and FName[-3]== "_": #version incrementee, numero a retirer pour trouver le ang original
            FName = FName[:-3]
        if FName[-7:]== "HexGrid": #version incrementee, numero a retirer pour trouver le ang original
            FName = FName[:-8] # et non -7 car on retire aussi le _
        print "**",FName, EBSDDirectory
        newrep = findfile(FName, os.path.join(EBSDDirectory, "maps_ANG"))
        if newrep=="?": newrep = findfile(FName, os.path.join(EBSDDirectory, "maps_TSL"))
        if GRID != "Hexagon":
            FName = FName.split('subset')[0]
            newrep = findfile(FName, os.path.join(EBSDDirectory, "maps_ANG"))
            if newrep=="?": newrep = findfile(FName, os.path.join(EBSDDirectory, "maps_TSL"))
            subset=1
            print "subset identified"
        else:
            FName = FName.split('_HexGridsubset')[0]
            newrep = findfile(FName, os.path.join(EBSDDirectory, "maps_ANG"))
            if newrep=="?": newrep = findfile(FName, os.path.join(EBSDDirectory, "maps_TSL"))
            subset=1
            print "subset and hexagonal grid identified"
        print "newfile=", FName, "new repertory",newrep
               
 
        if Transition == "Fe-NW" or Transition == "Fe-GT" or Transition == "Fe-G" or Transition == "Fe-KS" :
            phasetext = TSL_FeGamma
        elif Transition == "Burgers":
            phasetext = TSL_TiBeta
        elif Transition == "Zirconia-cubic-mono-type1" or Transition == "Zirconia-cubic-mono-type2":
            phasetext = TSL_ZrO2_cubic
        else:
            phasetext0 = TSL_FeGamma
            phasetext = ""
            for line in phasetext0:
                newline = line
                a = string.split(line)
                if "MaterialName" in a:
                    newline = "# MaterialName  	"+ str(CrystP.el)+"\n"
                elif "LatticeConstants" in a:
                    newline = "# LatticeConstants      "+cut(CrystP.a,5)+" "+ +cut(CrystP.b,5)+" "+cut(CrystP.c,5)+" "+cut(DEGREE*CrystP.alpha,5)+" "+cut(DEGREE*CrystP.beta,5)+" "+cut(DEGREE*CrystP.gamma,5)+"\n"
                elif "OPERATOR" in a or "SAMPLEID" in a or "SCANID" in a:
                    newline = ""
                phasetext = phasetext + newline


        os.chdir (newrep)
        bufile = open(FName + ".ang")
        i = 0
        Entete = ""
            
        for line in bufile :
            newline = line
            a = string.split(line)
            if ("#" not in a): break
            elif "Phase" in a: newline = "# Phase 0" + "\n" # the initial file is maintained, it is just the numbers of the phases that are all put at 0
            elif "NCOLS_ODD:" in a: newline = str("# NCOLS_ODD: " + str(Nbpixx+1) + "\n") 
            elif "NCOLS_EVEN:" in a: newline = str("# NCOLS_EVEN: " + str(Nbpixx) + "\n")
            elif "NROWS:" in a: newline = str("# NROWS: " + str(Nbpixy) + "\n")
            elif "OPERATOR" in a or "SAMPLEID" in a or "SCANID" in a: newline=""
            Entete = Entete + newline
            i+=1
            if i > 400: print "too much lines to define the file"
        bufile.close()
        Entete = Entete + phasetext # then I write the data of the parent phase
        text = "# OPERATOR:" + "\n" + "# " + "\n" + "# SAMPLEID: " + "\n" + "# " + "\n"  + "# SCANID: " + "\n" + "# "+"\n"
        Entete = Entete + "\n" + text

        
        z='000000'
        def compl(f,n=7):
            # complete f=1.2 en 1.20000 pour avoir 7 caracteres
            s=str(f)[:n]
            return s[:n]+z[:n-len(s)]
            
        p = EBSDDirectory + '/Results'
        os.chdir (p)
        fname = FileName + "_ParentGrains.ang"
        f = open(fname, 'w')
        f.write(Entete)
        X, Y = 0, 0
        Band = 6
        Error = 0
        C1 = '1000.1'
        C2 = '0.600'
        C4 = "1"
        kl = 0
        i=0
        for nb in range(len(TabPix)):
            X, Y = PosXY(nb)
            Xs,Ys = compl(X),compl(Y)
            E1 = TabParentGrains[TabPix[nb][4] - 1].euler1
            E2 = TabParentGrains[TabPix[nb][4] - 1].euler2
            E3 = TabParentGrains[TabPix[nb][4] - 1].euler3
            C2 = 1-float(TabGrains[TabPix[nb][3] - 1].confind)/30
            if C2<0: C2=0
            if C2==0:
                C2="0.000"
                C5="1.500"
                np = 0
            else:
                C2 = compl(C2)
                C5 = '1.500'
                np = 1
            s = '  %.7s   %.7s   %.7s      %.7s      %.7s %.7s  %s  %.3s  %s  %s\n' % (compl(E1/DEGREE), compl(E2/DEGREE), compl(E3/DEGREE), str(Xs), str(Ys), C1, compl(C2,5), np, C4, C5) 
            f.write(s)
            if GRID == "Hexagon":
                if (nb+1) % Nbpixx == 0:
                    if kl%2==0:
                        X = X + XStep
                        Xs = compl(X)
                        s = '  %.7s   %.7s   %.7s      %.7s      %.7s %.7s  %.3s  %s  %s  %s\n' % (compl(E1/DEGREE), compl(E2/DEGREE), compl(E3/DEGREE), str(Xs), str(Ys), C1, compl(C2,5), np, C4, C5)       
                        f.write(s) # pour recreer les odd et even
                    kl+=1
            i+=1
        f.close()
        EtatRun.set("\n"+"File " + fname + " created in " + os.path.join(EBSDDirectory, "Results")+"\n")
        LabelRun["fg"] = "blue"
        p = EBSDDirectory + "/Results"
        os.startfile(p)
        os.chdir (EBSDDirectory)
        
    except:
        try: exportParentToHKL()
        except: tkMessageBox.showinfo("ERROR:", 'Check there is no special letters in names of the folder tree and that the initial ctf file is located in the TSL folder. \n If the problem remains, contact C. Cayron')

#====================================================================================================  

def exportParentPixToTSL():
    
    if SourceFile == 'HKL':
        tkMessageBox.showinfo("Not possible:", 'the file of daughter grains was acquired with HKL')
        return
    if not TabPixP:
        tkMessageBox.showinfo("Not possible:", 'the pixel map of the parent grains must be calculated')
        return
    try:
        EtatRun.set("\n"+"Creating the file..."+"\n")
        LabelRun["fg"] = "red"
        frameFile.update()

        subset = 0
        FName = FileName
        if FName[-2]== "v" and FName[-3]== "_": #version incrementee, numero a retirer pour trouver le ang original
            FName = FName[:-3]
        if FName[-7:]== "HexGrid": #version incrementee, numero a retirer pour trouver le ang original
            FName = FName[:-8] # et non -7 car on retire aussi le _
        print "**",FName, EBSDDirectory
        newrep = findfile(FName, os.path.join(EBSDDirectory, "maps_ANG"))
        if newrep=="?": newrep = findfile(FName, os.path.join(EBSDDirectory, "maps_TSL"))
        if GRID != "Hexagon":
            FName = FName.split('subset')[0]
            newrep = findfile(FName, os.path.join(EBSDDirectory, "maps_ANG"))
            if newrep=="?": newrep = findfile(FName, os.path.join(EBSDDirectory, "maps_TSL"))
            subset=1
            print "subset identified"
        else:
            FName = FName.split('_HexGridsubset')[0]
            newrep = findfile(FName, os.path.join(EBSDDirectory, "maps_ANG"))
            if newrep=="?": newrep = findfile(FName, os.path.join(EBSDDirectory, "maps_TSL"))
            subset=1
            print "subset and hexagonal grid identified"
        print "newfile=", FName, "new repertory",newrep
               
        os.chdir (newrep)
        bufile = open(FName + ".ang")
        i = 0
        Entete = ""
 
        if Transition == "Fe-NW" or Transition == "Fe-GT" or Transition == "Fe-G" or Transition == "Fe-KS" :
            phasetext = TSL_FeGamma
        elif Transition == "Burgers":
            phasetext = TSL_TiBeta
        else: phasetext = "Parent phase"

        skip=0
        phaseid = 0
        hklfam = 0

        for line in bufile :
            a = string.split(line)
            if "Phase" in a:
                skip=1
            elif ("Categories0" in a) or ("Categories1" in a) or (hklfam ==2): #certains fichiers ang ne contiennent pas "categories"
                skip=0
                if not phaseid: text = phasetext+"\n"
                else: text=""
                phaseid = 1
                hklfam = 0
            elif "hklFamilies" in a and skip ==1:
                hklfam =1
            elif "hklFamilies" not in a and skip ==1 and hklfam ==1: # au cas ou "categories" pas ecrit dans fichier
                hklfam =2
            elif "ElasticConstants" in a: text = ""
            elif "OPERATOR:" in a: text = "# OPERATOR: \t ARPGE\n"
            elif "NCOLS_ODD:" in a: text = str("# NCOLS_ODD: " + str(Nbpixx+1) + "\n") 
            elif "NCOLS_EVEN:" in a: text = str("# NCOLS_EVEN: " + str(Nbpixx) + "\n")
            elif "NROWS:" in a: text = str("# NROWS: " + str(Nbpixy) + "\n")
            else : text = line

            if not skip : Entete = Entete + text

            if ("SCANID:" in a) or ("#" not in a):
                Entete = Entete + "#\n"
                break

            i+=1
            if i > 400: print "too much lines to define the file"
        bufile.close()

        z='000000'
        def compl(f):
            # complete f=1.2 en 1.20000 pour avoir 7 caracteres
            s=str(f)[:7]
            return s[:7]+z[:7-len(s)]
            
        newrep = EBSDDirectory + '/Results'
        os.chdir (newrep)
        fname = FileName + "_ParentPixels.ang"
        f = open(fname, 'w')
        f.write(Entete)
        X, Y = 0, 0
        Band = 6
        Error = 0
        C1 = '1000.1'
        C2 = '0.600'
        C4 = "1"
        kl = 0
        i=0
        for nb in range(len(TabPixP)):
            X, Y = PosXY(nb)
            Xs,Ys = compl(X),compl(Y)
            E1 = TabPixP[nb].euler1
            E2 = TabPixP[nb].euler2
            E3 = TabPixP[nb].euler3
            C2 = 1-float(TabGrains[TabPix[nb][3] - 1].confind)/30
            if C2<0: C2=0
            if C2==0:
                C2="-1.000"
                C5="180.000"
                np = -1
            else:
                C2 = compl(C2)
                C5 = '1.500'
                np = 1
            s = '  %.7s   %.7s   %.7s      %.7s      %.7s %.7s  %s  %d  %s  %s\n' % (compl(E1/DEGREE), compl(E2/DEGREE), compl(E3/DEGREE), str(Xs), str(Ys), C1, C2, np, C4, C5) 
            f.write(s)
            if GRID == "Hexagon":
                if (nb+1) % Nbpixx == 0:
                    if kl%2==0:
                        X = X + XStep
                        Xs = compl(X)
                        s = '  %.7s   %.7s   %.7s      %.7s      %.7s %.7s  %s  %d  %s  %s\n' % (compl(E1/DEGREE), compl(E2/DEGREE), compl(E3/DEGREE), str(Xs), str(Ys), C1, C2, np, C4, C5)       
                        f.write(s) # pour recreer les odd et even
                    kl+=1
            i+=1
        f.close()
        EtatRun.set("\n"+"File " + fname + " created in " + os.path.join(EBSDDirectory, "Results")+"\n")
        LabelRun["fg"] = "blue"
        p = EBSDDirectory + "/Results"
        os.startfile(p)
        os.chdir (EBSDDirectory)
    except:
        try: exportParentToTSL()
        except: tkMessageBox.showinfo("ERROR:", 'Check there is no special letters in names of the folder tree and that the initial ctf file is located in the TSL folder. \n If the problem remains, contact C. Cayron')
            

#====================================================================================================  
def exportPacketsToTSL(optexport):
    if SourceFile == 'HKL': tkMessageBox.showinfo("Not possible:", 'the file of daughter grains was acquired with HKL')
    else:
        tkMessageBox.showinfo("Done for .ctf files, not yet .ang files:", 'Please contact C. Cayron to make him finish the job !')
#====================================================================================================
def exportPacketsToHKL(optexport):
    #optexport = for packets, 1 = Bain packets, 2 = CPP packets, 3 = CPD packets
    if SourceFile == 'TSL': tkMessageBox.showinfo("Not possible:", 'the file of daughter grains was acquired with TSL')
    else: exportParentToHKL (optexport)
#====================================================================================================

def ShowExportPackets():
    ExportOpt = IntVar()
    ExportOpt.set(0)
    # 0 = USUAL APRENT GRAINS,  1 = Bain, 2 = Cryst Pack, 3 = CPD
    fExport = Toplevel(root)
    def exportPackets(event):
        print "option export"
        print ExportOpt.get()+1       
        if SourceFile == 'HKL' or SourceFile == "ARPGE_HKL" : exportPacketsToHKL(ExportOpt.get()+1)
        elif SourceFile == 'TSL' or SourceFile == "ARPGE_TSL": exportPacketsToTSL(ExportOpt.get()+1)
        elif SourceFile == "ARPGE" : tkMessageBox.showinfo("Not able to determine the original file format, .ctf or .ang", info)
            
    lt = Label(fExport, foreground="black")
    lt["text"] = "Export: the Band Slope (BS) column is replaced by the packet number in the exported .ctf file"
    lt.grid(row=1, column=1, columnspan=2, sticky=E + W)
    lt = Label(fExport, foreground="red")
    lt["text"] = "Be sure that the packets have been calculated by choosing Parent Grains / Show Packets or by cliking on the Bain, CPP or CPD button"
    lt.grid(row=2, column=1, columnspan=2, sticky=E + W)
    Radiobutton (fExport, text="Bain Packets", variable=ExportOpt, value=0).grid (row=3, column=1, sticky=W)
    Radiobutton (fExport, text="CPP Packets (Morito)", variable=ExportOpt, value=1).grid (row=4, column=1, sticky=W)
    Radiobutton (fExport, text="CPD Packets", variable=ExportOpt, value=2).grid (row=5, column=1, sticky=W)
    ButtonExp = Button(fExport, text="Export File", width=7, height=1, foreground="black", background="wheat")
    ButtonExp.bind("<Button>", exportPackets)
    ButtonExp.grid (row=6, column=1, columnspan=2)


#====================================================================================================  

def ExportCanvas ():
    # transforms the images of the map in the canvas into an tif image 
    # not used anymore

    nx = Nbpixx
    ny = Nbpixy
    di, dj = 1, 1
    p = EBSDDirectory + '/Results'
    os.chdir (p)
    extens = ''
    if OptShowB == "pixels": extens = "_Pixels"
    elif OptShowB == "graind": extens = "_DaughterGrains"
    elif OptShowB == "grainp": extens = "_ParentGrains"
    
    if OptShowA == "Triangle": extens = "_DG_RGB1"
    elif OptShowA == "Triangle2": extens = "_DG_RGB2"

##    oname = asksaveasfilename(initialdir=p, title='Save as ...',filetypes=[('bmp files','*.bmp')])
##    if not oname: oname = p+"/"+FileName+extens+'.bmp'
##    if ".bmp" not in oname: oname = oname+".bmp"

    oname = os.path.join(p, FileName + extens)
    draw = ImageDraw.ImageDraw(image)


    # VarNum
    if VarNum.get():
        oname = oname + '_VarNum'
        if TabParentGrains:
            for grp in TabParentGrains:
                for indgr in grp.graind:
                    gr = TabGrains[indgr - 1]
                    if Transition != "Sigma3":
                        if gr.variant: #n affiche que si un seul numero est trouve
                            wd = str(gr.variant[0])
                            if GRID != "Hexagon": x, y = int((gr.x / Xmax) * nx * Scale), int((gr.y / Ymax) * ny * Scale)
                            else: x, y = int((gr.x / Xmax) * nx * Scale), int((gr.y / Ymax) * ny * Scale * (sqrt(3) / 2))
                            if len(wd) == 1:
                                draw.rectangle((x - 3, y - 5, x + 3, y + 5), fill="white")
                                draw.text((x - 2, y - 5), wd, fill="black", anchor = "center")
                            else:
                                draw.rectangle((x - 6, y - 5, x + 6, y + 5), fill="white")
                                draw.text((x - 6, y - 5), wd, fill="black",anchor = "center")

    # GrainNum
    if OptShowN and OptShowB == "graind" and TabGrains:
        oname = oname + '_GrainNum'
        for gr in TabGrains:
            wd = str(gr.graind)
            if GRID != "Hexagon": x, y = int((gr.x / Xmax) * nx * Scale), int((gr.y / Ymax) * ny * Scale)
            else: x, y = int((gr.x / Xmax) * nx * Scale), int((gr.y / Ymax) * ny * Scale*sqrt(3)/2)
            draw.rectangle((x - 5*len(wd), y - 5, x + 5*len(wd), y + 5), fill="white")
            draw.text((x - 4, y - 5), wd, fill="black",anchor = "center")

    elif OptShowN and OptShowB == "grainp" and TabParentGrains:
        oname = oname + '_GrainNum'
        for gr in TabParentGrains:
            wd = str(gr.grainp)
            if GRID != "Hexagon": x, y = int((gr.x / Xmax) * nx * Scale), int((gr.y / Ymax) * ny * Scale)
            else: x, y = int((gr.x / Xmax) * nx * Scale), int((gr.y / Ymax) * ny * Scale*sqrt(3)/2)
            draw.rectangle((x - 5*len(wd), y - 5, x + 5*len(wd), y + 5), fill="white")
            draw.text((x-5*len(wd)/2, y - 5), wd, fill="black",anchor = "center")

    
    # Daughter GBs
    if DaughterGB.get():
        if TabGrains:
            for gr in TabGrains:
                for pix in gr.listpixfr:
                    i = pix / nx
                    j = pix - nx * i
                    if GRID != "Hexagon": draw.rectangle((int(j * Scale), int(i * Scale), int((j + dj) * Scale), int((i + di) * Scale)), fill="black")
                    else: draw.rectangle((int(j * Scale), int(i * Scale * sqrt(3) / 2), int((j + dj) * Scale), int((i + di) * Scale * sqrt(3) / 2)), fill="black")

    # Parent Retained GBs
    if ParentGB.get():
        if TabRetainedGrains:
            for gr in TabRetainedGrains:
                gbd = []
                di,dj = 2,2
                for pix in gr.listpixfr:
                    i = pix / nx
                    j = pix - nx * i
                    if GRID != "Hexagon": draw.rectangle((int(j * Scale), int(i * Scale), int((j + dj) * Scale), int((i + di) * Scale)), fill="black")
                    else: draw.rectangle((int(j * Scale), int(i * Scale * sqrt(3) / 2), int((j + dj) * Scale), int((i + di) * Scale * sqrt(3) / 2)), fill="black")
    # Parent GBs
    if ParentGB.get():
        if TabParentGrains:
            for gr in TabParentGrains:
                gbd = []
                di,dj = 2,2
                for pix in gr.listpixfr:
                    i = pix / nx
                    j = pix - nx * i
                    if GRID != "Hexagon": draw.rectangle((int(j * Scale), int(i * Scale), int((j + dj) * Scale), int((i + di) * Scale)), fill="black")
                    else: draw.rectangle((int(j * Scale), int(i * Scale * sqrt(3) / 2), int((j + dj) * Scale), int((i + di) * Scale * sqrt(3) / 2)), fill="black")
    

    # Scale bar
    dLmin, L = Nbpixx,Nbpixx
    listL = [0.01,0.05,0.1,0.2,0.5,1,2,5,10,20,30,40,50,100,200,300,400,500,1000,2000,3000,4000,5000,10000]
    for Li in listL:
        ix = int(Li/XStep)
        dL = abs(Nbpixx/3-ix)
        if dL<dLmin:
            L = Li
            dLmin = dL
    ScaleBar = int(L / XStep)
  
    try: font = ImageFont.truetype("arial.ttf", 10)
    except: font =ImageFont.load_default() 
    draw.rectangle((0, image.size[1] - 1, ScaleBar*Scale + 10, image.size[1] - 30), fill="white")
    draw.line((3, image.size[1] - 3, ScaleBar*Scale + 3, image.size[1] - 3), fill="black")
    t = str(L) + " " + Units
    try: draw.text((3+(ScaleBar*Scale-len(t)*10)/2, int(image.size[1]-20)), t, font=font, fill="black", anchor = "center")
    except:
        try: 
            font = ImageFont.truetype ('C:\WINDOWS\Fonts\Arial.ttc', 10)
            draw.text((3+(ScaleBar*Scale-len(t)*10)/2, int(image.size[1]-20)), t, font=font, fill="black", anchor = "center")
        except:
            try:
                font = ImageFont.truetype ('C:\WINDOWS\Fonts\Arial.ttf', 10)
                draw.text((3+(ScaleBar*Scale-len(t)*10)/2, int(image.size[1]-20)), t, font=font, fill="black", anchor = "center")
            except:
                print "pb : font not found. Check that C:\WINDOWS\Fonts exists"

    oname = oname + '.bmp'
    image.save(oname, "bmp")

    EtatRun.set("\n"+"Image saved in " + p+"\n")
    LabelRun["fg"] = "blue"
    os.startfile(p)
    os.chdir (EBSDDirectory)

#====================================================================================================  

def SaveCanvas (event):
    # new method: directly save the canvas
    if ImageSaveMode.get()==1:
        d = EBSDDirectory + '/Results/'
        os.chdir (d)
        extens = ''
        
        if OptShowB == "pixels": extens = "_Pixels"
        elif OptShowB == "graind": extens = "_DaughterGrains"
        elif OptShowB == "grainp": extens = "_ParentGrains"
        
        if OptShowA == "Triangle": extens = "_DG_RGB1"
        elif OptShowA == "Triangle2": extens = "_DG_RGB2"
        elif OptShowA == "gos": extens = "_GOS_"+"mean="+cut(GOSmean)+"deg"+"_max="+cut(gosmax)+"deg"
        elif OptShowA == "def": extens = "_KAM_"+"mean="+cut(KAMmean)+"deg"+"_max="+cut(defmax)+"deg"
        elif OptShowA == "def2": extens = "_KAMmean_"+"mean="+cut(KAMmean)+"deg"+"_max="+cut(defmax)+"deg"

        filename = d + FileName + extens

        WC,HC = canvas.winfo_width(), canvas.winfo_height()
        x1,y1 = canvas.winfo_rootx(), canvas.winfo_rooty()
        ImageGrab.grab((x1,y1,x1+WC,y1+HC)).save(filename + '.bmp')
        tkMessageBox.showinfo("info", "the histogram is saved in the directory: " + d)
        os.startfile(d)
    else:
        ExportCanvas()
    
#====================================================================================================  

def HKLToTSL():
    
    if len(CrystD.sym()) == 48: sym = 43 # je ne comprends rien au codage de TSL
    else : sym = len(CrystD.sym())
    if SourceFile == 'TSL':
        tkMessageBox.showinfo("Attention please:", 'the load file is already a TSL file!')
    os.chdir (EBSDDirectory + '\Results')
    Entete = ["# TEM_PIXperUM \t 1.000000",
             "\n" + "# x-star \t 0.5",
             "\n" + "# y-star \t 0.6",
             "\n" + "# z-star \t 0.5",
             "\n" + "# WorkingDistance \t 15.000000",
             "\n" + "# ",
             "\n" + "# Phase 1" + "\n" + "# MaterialName \t" + 	CrystD.el,
             "\n" + "# Formula \t" + 	CrystD.el,
             "\n" + "# Info" ,
             "\n" + "# Symmetry \t" + str(sym),
             "\n" + "# LatticeConstants \t" + str(CrystD.a) + ' ' + str(CrystD.b) + ' ' + str(CrystD.c) + ' ' + str(CrystD.alpha * DEGREE) + ' '
             + str(CrystD.beta * DEGREE) + ' ' + str(CrystD.gamma * DEGREE),
             "\n" + "# NumberFamilies  \t 1",
             "\n" + "# hklFamilies \t 1  1  0 1 0.000000 1",
             "\n" + "# hklFamilies \t 2  0  0 1 0.000000 1",
             "\n" + "# hklFamilies \t 2  1  1 1 0.000000 1",
             "\n" + "# hklFamilies \t 3  1  0 1 0.000000 1",
             "\n" + "# Categories0 0 0 0 0",
             "\n" + "# ",
             "\n" + "# GRID: SqrGrid",
             "\n" + "# XSTEP: " + str(XStep),
             "\n" + "# YSTEP: " + str(YStep),
             "\n" + "# NCOLS_ODD: " + str(Nbpixx),
             "\n" + "# NCOLS_EVEN: " + str(Nbpixx),
             "\n" + "# NROWS: " + str(Nbpixy),
             "\n" + "# ",
             "\n" + "# OPERATOR \t from HKL file via ARPGE",
             "\n" + "# ",
             "\n" + "# SAMPLEID:",
             "\n" + "# ",
             "\n" + "# SCANID:",
             "\n" + "# " + "\n"]
    Entete = add.reduce(Entete)

    fname = FileName + "ARPGE.ang"
    f = open(fname, 'w')
    f.write(Entete)
##    print Nbpixx,Nbpixy, XStep
    X, Y = 0, 0
    truc = 0.8
    Error = 0
    MAD = 0.1
    BC = 150
    BS = 0.4
    XMax = Nbpixx * XStep
    YMax = Nbpixy * YStep
    lentab = len(TabPix)
    stepg = lentab/StProg
    for nb in range(lentab):
        if not nb%stepg: UpDateProgress((nb*100)/lentab)
        E1 = TabPix[nb][0] / DEGREE
        E2 = TabPix[nb][1] / DEGREE
        E3 = TabPix[nb][3] / DEGREE
        s = '%3.4f \t %3.4f \t %3.4f \t %3.4f \t %3.4f \t %3.4f \t %3.4f \t %1d \t %3d \t %1.4f \n' % (E1, E2, E3, X, Y, BC, MAD, 1, 1000, truc)
        X = X + XStep
        if X >= XMax:
            X = 0
            Y = Y + YStep
        f.write(s)
    f.close()
    print "file .ang created"
    os.chdir (EBSDDirectory)

    EtatRun.set("\n"+"file " + fname + " created"+"\n")
    LabelRun["fg"] = "blue"
    UpDateProgress(0)
    #frameFile.update()

#====================================================================================================  

def TSLToHKL():
    pass

#===================================================================================================

def create_SubSet0():
    
    global OptShowC
    OptShowC = "Rectangle"
    EtatRun.set("\n"+"In the graphic window, draw a rectangle by click & drag with mouse"+"\n")
    LabelRun["fg"] = "red"

#=================================================================================================== 

def create_SubSet():
    
    global Longueur, Nbpixx, Nbpixy, TimeStart
    global euler1min, euler2min, euler3min, deuler1, deuler2, deuler3, bcmin, dbc, FileName, Scale
    global euler1minP, euler2minP, euler3minP, deuler1P, deuler2P, deuler3P
    global Xmax, Ymax, Entete
    global TabPix, TabGrains, TabPixP
    global OptShowC

    frameFile.update()

    TabGrains = []
    FileName = FileName + "subset0"

    euler1min,euler1max,euler2min,euler2max,euler3min,euler3max = 360,0,360,0,360,0
    euler1minP,euler1maxP,euler2minP,euler2maxP,euler3minP,euler3maxP = 360,0,360,0,360,0
##    bcmin = 255
##    bcmax = 0
    Xmax = 0
    Ymax = 0
    Xmin = 10000000000
    Ymin = 10000000000

    iStart, jStart = ijFromxy(xStart, yStart)
    iFinish, jFinish = ijFromxy(xFinish, yFinish)

    tabpixnew = []
    for i in range(iStart, iFinish):
        for j in range(jStart, jFinish):
            pix = TabPix[i * Nbpixx + j]
            tabpixnew.append(pix)
            x = j * XStep
            y = i * YStep
            if x > Xmax : Xmax = x
            if y > Ymax : Ymax = y
            if x < Xmin : Xmin = x
            if y < Ymin : Ymin = y
            if pix[0] < euler1min : euler1min = pix[0]  #euler1
            elif pix[0] > euler1max : euler1max = pix[0]
            if pix[1] < euler2min : euler2min = pix[1]  #euler2
            elif pix[1] > euler2max : euler2max = pix[1]
            if pix[2] < euler3min : euler3min = pix[2]  #euler3
            elif pix[2] > euler3max : euler3max = pix[2]
##            if pix[5] < bcmin : bcmin = pix[5]  #bc
##            elif pix[5] > bcmax : bcmax = pix[5]

    deuler1 = euler1max - euler1min
    deuler2 = euler2max - euler2min
    deuler3 = euler3max - euler3min
##    dbc     = bcmax     - bcmin
    TabPix = tabpixnew

    if len(TabPixP)>0:
        tabpixnewP = []
        for i in range(iStart, iFinish):
            for j in range(jStart, jFinish):
                pix = TabPixP[i * Nbpixx + j]
                tabpixnewP.append(pix)
                x = j * XStep
                y = i * YStep
                if x > Xmax : Xmax = x
                if y > Ymax : Ymax = y
                if x < Xmin : Xmin = x
                if y < Ymin : Ymin = y
                if pix[0] < euler1minP : euler1minP = pix[0]  #euler1
                elif pix[0] > euler1maxP : euler1maxP = pix[0]
                if pix[1] < euler2minP : euler2minP = pix[1]  #euler2
                elif pix[1] > euler2maxP : euler2maxP = pix[1]
                if pix[2] < euler3minP : euler3minP = pix[2]  #euler3
                elif pix[2] > euler3maxP : euler3maxP = pix[2]

        deuler1P = euler1maxP - euler1minP
        deuler2P = euler2maxP - euler2minP
        deuler3P = euler3maxP - euler3minP

        TabPixP = tabpixnewP

    Longueur = len(tabpixnew)
    Nbpixx = abs(jFinish - jStart)
    Nbpixy = abs(iFinish - iStart)
    print "Nbpixx ,Nbpixy = ", Nbpixx, Nbpixy 

    Xmax = Xmax - Xmin
    Ymax = Ymax - Ymin

    if GRID == "Hexagon": Ymax = Ymax * sqrt(3)/2

    NewEntete = ""
    Lines = Entete.split("\n")
    for line in Lines:
        if "XCells" in line : nline = "XCells\t" + str(Nbpixx)
        elif "YCells" in line : nline = "YCells\t" + str(Nbpixy)
        else: nline = line 
        NewEntete = NewEntete + nline + '\n'
    Entete = NewEntete[:-1] # pour retirer le dernier retour a la ligne

    #TabPix = tabpixnew

    EtatFile.set(FileName)
    EtatSize.set(str(Nbpixx) + " x " + str(Nbpixy) + " pixels")

    EtatRun.set("\n"+"Next step: filter the pixels and identify the daughter grains"+"\n")
    LabelRun["fg"] = "blue"
    os.chdir(EBSDDirectory)
    if not HideFlag.get(): showmap()
    OptShowC = "Line"

#===================================================================================================

def openGrainFile():
    
    global Longueur, Nbpixx, Nbpixy, XStep, YStep, deuler1, deuler2, deuler3, euler1min, euler2min, euler3min, bcmin, dbc
    global Transition, Scale, SourceFile
    global Xmax, Ymax, FileName
    global TabPix, TabGrains, TabParentGrains
    global FlagRec, GRID
    global TransLoadMode
    global NextStep, ListPlanes
    global Entete

    TransLoadMode = 1
    TabPix = []
    TabGrains = []
    TabParentGrains = []
    Entete = ""
    
    os.chdir (os.path.join(EBSDDirectory, "Results"))
    oname = askopenfilename(filetypes=[("text files", "*.arpge")])
    if oname :
        EtatRun.set("\n"+"Opening the file..."+"\n")
        LabelRun["fg"] = "red"
        frameFile.update()
        FileName = oname
        FileName = str(FileName[:len(FileName) - 6])
        FileName = FileName[FileName.find("Results/") + 8:]
        #print "FileName",FileName
        if "HexGrid" in FileName: GRID = "Hexagon"

        LoadResults()

        os.chdir(EBSDDirectory)

        Longueur = len(TabPix)

        EtatFile.set(FileName)
        EtatSize.set(str(Nbpixx) + " x " + str(Nbpixy) + " pixels")
        EtatTrans.set(Transition)

        if Transition == "Steel NW" : Transition = "Fe-NW"
        if Transition == "Fe-NW" or Transition == "Fe-GT" or Transition == "Fe-G" or Transition == "Fe-KS" : ListPlanes = ListPlanesCubic
        elif Transition == "Burgers": ListPlanes = ListPlanesHex

        if Transition != "Sigma3": LoadTransitionFiles()
        else :
            ListPlanes = ListPlanesCubic
            LoadTransitionFilesSigma3()

        os.chdir (os.path.join(EBSDDirectory, "Results"))
        NextStep = 5

    else:
        EtatRun.set("\n"+"Cannot open the file!"+"\n")
        frameFile.update()

    EtatRun.set("\n"+"Next step: Suppress the Small Grains or Reconstruct the Parent Grains"+"\n")
    LabelRun["fg"] = "blue"
    os.chdir(EBSDDirectory)

    #print "Next Step", NextStep
    if not HideFlag.get(): showmap()

#===================================================================================================

def palette():
    
    # converti 0<nb<255 en un triplet RGB
    
    gamma = 0.98

    def factorcol(f):
        if 380 <= f < 420:
            fact = 0.3 + 0.7 * (f - 380) / (420 - 380)
        elif 420 <= f < 700:
            fact = 1.
        elif 700 <= f <= 780:
            fact = 0.3 + 0.7 * (780 - f) / (780 - 700)
        else: fact = 0.
        return fact

    def adjust(col, factor):
        if col == 0: r = 0
        else: r = int(255 * (col * factor) ** gamma)
        return r

    p = []
    for i in range(256):
        #f = 380 + (float(i) / 255) * (780 - 380) #frequence du visible
        f = 420 + (float(i) / 255) * (700 - 380) #frequence du visible
        if 380 <= f < 440:
            r = (440 - f) / (440 - 380)
            g = 0.
            b = 1.
        elif 440 <= f < 490:
            r = 0.
            g = (f - 440) / (490 - 440)
            b = 1.
        elif 490 <= f < 560: #560 avant 510
            r = 0.
            g = 1.
            b = (560 - f) / (560 - 490)
        elif 560 <= f < 610: # 610 avant 580
            r = (f - 560) / (610 - 560)
            g = 1.
            b = 0.
        elif 610 <= f < 700: # 700 avant 645
            r = 1.
            g = (700 - f) / (700 - 610)
            b = 0.
        elif 700 <= f <= 780:
            r = 1.
            g = 0.
            b = 0.
        a = factorcol(f)
        r = adjust(r, a)
        g = adjust(g, a)
        b = adjust(b, a)
        rgb = (r, g, b)
        p.append(rgb)
    return p

#===================================================================================================

def showmap():
    
    global OptShowA,OptShowB, imageTk, imageCV, image, canvas, ScaleBar, Units
    global Units

    infoRet = ""

    if OptShowA == "" :
        textRun = EtatRun.get()
        EtatRun.set("\n"+"Preparing the graphics..."+"\n")
        LabelRun["fg"] = "red"
    elif OptShowA == "IPF":
        EtatRun.set("\n"+"Calculating (quite long, sorry)..."+"\n")
        LabelRun["fg"] = "red"
        LabelRun.update()
    if OptShowRet ==0: infoRet = "Reconstructed Parent Grains."
    elif OptShowRet ==1: infoRet = "Retained Parent Grains." 
    elif OptShowRet ==2: infoRet = "Reconstructed + Retained Parent Grains."
    if OptShowB == "grainp":
        EtatRun.set("\n"+infoRet+"\n")
        LabelRun["fg"] = "blue"
        LabelRun.update()
        

    all_image = canvas.find_withtag("image")
    for im in all_image: canvas.delete(im)

    nx = Nbpixx
    ny = Nbpixy

    euler1minP, euler2minP, euler3minP = 360, 360, 360
    euler1maxP, euler2maxP, euler3maxP = 0, 0, 0
    deuler1P, deuler2P, deuler3P = 360, 360, 360
    confmin, dconf = 0,0

    image = Image.new("RGB", (nx, ny), "white")
    #if GRID== "Hexagon": image = Image.new("RGB",(nx,ny+nx),"white")
    draw = ImageDraw.ImageDraw(image)

    i = 0

    if OptShowA == "euler1":
        EMin.set(cut(euler1min))
        EMax.set(cut(euler1min + deuler1)+" Deg.")
    elif OptShowA == "euler2":
        EMin.set(cut(euler2min))
        EMax.set(cut(euler2min + deuler2)+" Deg.")
    elif OptShowA == "euler3":
        EMin.set(cut(euler3min))
        EMax.set(cut(euler3min + deuler3)+" Deg.")
    elif OptShowA == "euler123":
        EMin.set(cut((euler1min + euler2min + euler3min) / 3.))
        EMax.set(cut((euler1min + euler2min + euler3min + deuler1 + deuler2 + deuler3) / 3.)+" Deg.")
    elif OptShowA == "IPF":
        EMin.set("0")
        if modeIPF==2: EMax.set("IPF_Z")
        elif modeIPF==1: EMax.set("IPF_Y")
        elif modeIPF==0: EMax.set("IPF_X")
        invTSD = inv(TSD)
        invTSP = inv(TSP)
        if modeIPF==2: axeS = [0,0,1]
        elif modeIPF==1: axeS = [1,0,0] # a voir pourquoi il faut interchager X et Y
        elif modeIPF==0: axeS = [0,1,0]
    elif OptShowA == "qual" and OptShowB == "pixels":
        EMin.set(cut(bcmin))
        EMax.set(cut(bcmin+dbc)) # grey scale
    elif OptShowA == "qual" and OptShowB == "graind":
        lconf = [TabGrains[u].confind for u in range(len(TabGrains))]
        confmin = min(lconf)
        confmax = max(lconf)
        dconf = confmax-confmin
        EMin.set(cut(confmin))
        EMax.set(cut(confmax)+" Deg.")
    elif OptShowA == "qual" and OptShowB == "grainp":
        lconf = [TabParentGrains[u].confind for u in range(len(TabParentGrains))]
        confmin = min(lconf)
        confmax = max(lconf)
        dconf = confmax-confmin
        EMin.set(cut(confmin))
        EMax.set(cut(confmax)+" Deg.")
    elif OptShowA == "numbers" and not TabGrains and not TabRetainedGrains:
        EMin.set(0.)
        EMax.set(0.)
    elif OptShowA == "numbers" and OptShowB == "graind" :
        lenTab = len(TabGrains)
        EMin.set(1.)
        EMax.set(float(lenTab))
        RandList = random.sample(range(lenTab), lenTab)
##    elif OptShowA == "numbers" and not TabParentGrains:
##        EMin.set(0.)
##        EMax.set(0.)
    elif OptShowA == "numbers" and OptShowB == "grainp" :
        lenTab = len(TabParentGrains)+len(TabRetainedGrains)
        EMin.set(1)
        EMax.set(int(lenTab))
        RandList = random.sample(range(lenTab), lenTab)
    elif OptShowA == "def":
        EMin.set(cut(defmin))
        EMax.set(cut(defmax)+" Deg.")
    elif OptShowA == "gos":
        EMin.set(cut(gosmin))
        EMax.set(cut(gosmax)+" Deg.")
    elif OptShowA == "Triangle": 
        EMin.set("0")
        EMax.set("10 Deg. in RGB")
    elif OptShowA == "Triangle2": 
        EMin.set("0")
        EMax.set("2.5 Deg. Red = Pitsch, Green = KS, Blue = NW, Orange = G")
    elif OptShowA == "def2":
        kammax = 0
        kammin = 90
        for grd in TabGrains:
            if grd.kam> kammax:  kammax = grd.kam
            if grd.kam< kammin:  kammin = grd.kam
        EMin.set(cut(kammin)+" Deg.")
        EMax.set(cut(kammax)+" Deg.")
    elif OptShowA != "numbers" and (OptShowB == "grainp" or OptShowA == "pixelsP"):
        if OptShowB == "grainp" and len(TabParentGrains)>0:
            lenTab = len(TabParentGrains)
            for grp in TabParentGrains:
                e1P, e2P, e3P = grp.euler1, grp.euler2, grp.euler3
                if e1P < euler1minP: euler1minP = e1P
                if e2P < euler2minP: euler2minP = e2P
                if e3P < euler3minP: euler3minP = e3P
                if e1P > euler1maxP: euler1maxP = e1P
                if e2P > euler2maxP: euler2maxP = e2P
                if e3P > euler3maxP: euler3maxP = e3P
        else:
            for pixP in TabPixP:
                e1P, e2P, e3P = pixP
                if e1P < euler1minP: euler1minP = e1P
                if e2P < euler2minP: euler2minP = e2P
                if e3P < euler3minP: euler3minP = e3P
                if e1P > euler1maxP: euler1maxP = e1P
                if e2P > euler2maxP: euler2maxP = e2P
                if e3P > euler3maxP: euler3maxP = e3P
        deuler1P, deuler2P, deuler3P = euler1maxP - euler1minP, euler2maxP - euler2minP, euler3maxP - euler3minP
        if deuler1P == 0: deuler1P = 1
        if deuler2P == 0: deuler2P = 1
        if deuler3P == 0: deuler3P = 1
        if OptShowA == "euler1":
            EMin.set(cut(euler1minP))
            EMax.set(cut(euler1maxP))
        elif OptShowA == "euler2":
            EMin.set(cut(euler2minP))
            EMax.set(cut(euler2maxP))
        elif OptShowA == "euler3":
            EMin.set(cut(euler3minP))
            EMax.set(cut(euler3maxP))
        elif OptShowA == "euler123":
            EMin.set(cut((euler1minP + euler2minP + euler3minP) / 3.))
            EMax.set(cut((euler1maxP + euler2maxP + euler3maxP) / 3.))
        elif OptShowA == "IPF":
            EMin.set("0")
            if modeIPF==2: EMax.set("IPF_Z")
            elif modeIPF==1: EMax.set("IPF_Y")
            elif modeIPF==0: EMax.set("IPF_X")


    def CalcIPF(e1,e2,e3,invTS,cryst,listpix,IPFaxes):
        m = EulersToMat(e1,e2,e3,SourceFile) #m = [CrystOrtho->Samp]
        m = dot(invTS,m)
        Zc = dot(m,array(axeS))
        rgb = IPFCoord(Zc,IPFaxes,cryst)
        for ind in listpix:
            i = ind / Nbpixx
            j = ind - (i * Nbpixx)
            draw.point([(j, i)], fill=rgb)

    if OptShowA == "def":
        ddef = defmax-defmin
        while i < ny:
            for j in range(nx):
                ang = TabPixDef[i * nx + j]
                try: color = int(((ang - defmin) / ddef) * NbColors)
                except: pass 
                try: draw.point([(j, i)], fill=Palette[color])
                except: color = 0
            i+=1

    if OptShowA == "gos":
        dgos = gosmax-gosmin
        while i < ny:
            for j in range(nx):
                try: indgrd = TabPix[i * nx + j][3] - 1 #3 = graind
                except: indgrd =-1
                if indgrd != -1:
                    grd = TabGrains[indgrd]
                    gos = grd.gos
                    try: color = int(((gos - gosmin) / dgos) * NbColors)
                    except: pass 
                    try: draw.point([(j, i)], fill=Palette[color])
                    except: color = 0
            i+=1
            
    elif OptShowA == "CPDPackets":
        while i < ny:
            for j in range(nx):
                try: indgr = TabPix[i * nx + j][3] - 1 #3 = graind
                except: indgr =-1
                if indgr != -1:
                    try:
                        gr = TabGrains[indgr]
                        v = gr.variant[0]
                        if v:
                            if v == 1 or v == 3 or v == 4 or v == 9: color = "blue"
                            elif v == 5 or v == 6 or v == 16 or v == 17: color = "green"
                            elif v == 11 or v == 12 or v == 13 or v== 23: color = "red"
                            elif v == 7 or v == 14 or v == 18 or v== 22: color = "gold"
                            elif v == 19 or v == 20 or v == 21 or v== 24: color = "pink"
                            elif v == 2 or v == 8 or v == 10 or v== 15: color = "chocolate"
                        else: color = "white"
                    except : color = "white"
                    try: draw.point([(j, i)], fill=color)
                    except: color = 0
                else: color = "white"
            i+=1

    elif OptShowA == "CPPPackets":
        while i < ny:
            for j in range(nx):
                try: indgr = TabPix[i * nx + j][3] - 1 #3 = graind
                except: indgr =-1
                if indgr != -1:
                    try:
                        gr = TabGrains[indgr]
                        v = gr.variant[0]
                        if v:
                            if v == 1 or v == 2 or v == 3 or v == 5 or v == 6 or v == 8 : color = "blue"
                            elif v == 4 or v == 7 or v == 9 or v == 11 or v == 12 or v == 14: color = "green"
                            elif v == 10 or v == 13 or v == 15 or v == 19 or v == 20 or v == 23: color = "red"
                            elif v == 16 or v == 17 or v == 18 or v == 21 or v == 22 or v == 24: color = "gold"
                        else: color = "white"
                    except : color = "white"
                    try: draw.point([(j, i)], fill=color)
                    except: color = 0
                else: color = "white"
            i+=1

    elif OptShowA == "BainPackets":
        while i < ny:
            for j in range(nx):
                try: indgr = TabPix[i * nx + j][3] - 1 #3 = graind
                except: indgr =-1
                if indgr != -1:
                    try:
                        gr = TabGrains[indgr]
                        v = gr.variant[0]
                        if v:
                            if v == 1 or v == 4 or v == 6 or v == 12 or v == 17 or v == 19 or v == 21 or v == 23 : color = "blue"
                            elif v == 5 or v == 8 or v == 11 or v == 13 or v == 14 or v == 15 or v == 16 or v == 22 : color = "green"
                            elif v == 2 or v == 3 or v == 7 or v == 9 or v == 10 or v == 18 or v == 20 or v == 24 : color = "red"
                        else: color = "white"
                    except : color = "white"
                    try: draw.point([(j, i)], fill=color)
                    except: color = 0
                else: color = "white"
            i+=1
            
    elif OptShowA == "BurstPackets":
        while i < ny:
            for j in range(nx):
                try: indgr = TabPix[i * nx + j][3] - 1 #3 = graind
                except: indgr =-1
                if indgr != -1:
                    try:
                        gr = TabGrains[indgr]
                        v = gr.variant[0]
                        if v:
                            if v == 1 or v == 8 or v == 21 or v == 22: color = "blue"
                            elif v == 2 or v == 6 or v == 7 or v == 12: color = "green"
                            elif v == 3 or v == 5 or v == 20 or v== 13: color = "red"
                            elif v == 4 or v == 14 or v == 19 or v== 15: color = "gold"
                            elif v == 9 or v == 11 or v == 24 or v== 16: color = "pink"
                            elif v == 10 or v == 23 or v == 18 or v== 17: color = "chocolate"
                        else: color = "white"
                    except : color = "white"
                    try: draw.point([(j, i)], fill=color)
                    except: color = 0
                else: color = "white"
            i+=1
      

    elif OptShowA == "packets010" and (Transition=="Zirconia-cubic-mono-type1" or Transition=="Zirconia-cubic-mono-type2") and TabParentGrains: #oct2017
        while i < ny:
            for j in range(nx):
                try: indgr = TabPix[i * nx + j][3] - 1 #3 = graind
                except: indgr =-1
                if indgr != -1:
                    try:
                        gr = TabGrains[indgr]
                        v = gr.variant[0]
                        if v:
                            if v == 1 or v == 4 or v == 7 or v == 11: color = "blue"
                            elif v == 3 or v == 5 or v == 8 or v == 10: color = "green"
                            elif v == 2 or v == 6 or v == 9 or v == 12: color = "red"
                        else: color = "white"
                    except :
                        color = "white"
                    try: draw.point([(j, i)], fill=color)
                    except: color = 0
                else: color = "white"
            i+=1

    elif OptShowA == "packets100" and (Transition=="Zirconia-cubic-mono-type1" or Transition=="Zirconia-cubic-mono-type2") and TabParentGrains: #oct2017
        while i < ny:
            for j in range(nx):
                try: indgr = TabPix[i * nx + j][3] - 1 #3 = graind
                except: indgr =-1
                if indgr != -1:
                    try:
                        gr = TabGrains[indgr]
                        v = gr.variant[0]
                        if v:
                            if v == 1 or v == 4 or v == 6 or v == 12: color = "blue"
                            elif v == 5 or v == 7 or v == 10 or v == 11: color = "green"
                            elif v == 2 or v == 3 or v == 8 or v == 9: color = "red"
                        else: color = "white"
                    except :
                        color = "white"
                    try: draw.point([(j, i)], fill=color)
                    except: color = 0
                else: color = "white"
            i+=1
    
    elif OptShowA == "packets001" and (Transition=="Zirconia-cubic-mono-type1" or Transition=="Zirconia-cubic-mono-type2") and TabParentGrains: #oct2017
        while i < ny:
            for j in range(nx):
                try: indgr = TabPix[i * nx + j][3] - 1 #3 = graind
                except: indgr =-1
                if indgr != -1:
                    try:
                        gr = TabGrains[indgr]
                        v = gr.variant[0]
                        if v:
                            if v == 1 or v == 3 or v == 4 or v == 8: color = "blue"
                            elif v == 5 or v == 6 or v == 10 or v == 12: color = "green"
                            elif v == 2 or v == 7 or v == 9 or v == 11: color = "red"
                        else: color = "white"
                    except :
                        color = "white"
                    try: draw.point([(j, i)], fill=color)
                    except: color = 0
                else: color = "white"
            i+=1

    elif OptShowA == "Triangle" and TabPixTriangle:
        while i < ny:
            for j in range(nx):
                try:
                    triang = TabPixTriangle[i * nx + j] # = spread angles between exp and closest theor. variant (Pitsch, KS, NW)
                    if triang[0]<10 and triang[1]<10 and triang[2]<10 :
                        r,g,b = int(255-triang[0]*255/10), int(255-triang[1]*255/10), int(255-triang[2]*255/10) # 10 = degres ang max, max dans le triable P-NW-KS
                        rgb = (r,g,b)
                    else: rgb = "black"
                except: rgb = "black"
                draw.point([(j, i)], fill=rgb)
            i+=1
    
    elif OptShowA == "Triangle2" and TabPixTriangle:
        ct, c0,c1,c2,c3 = 0,0,0,0,0
        while i < ny:
            for j in range(nx):
                try:
                    triang = TabPixTriangle[i * nx + j] # = spread angles between exp and closest theor. variant (Pitsch, KS, NW)

                    if triang[0]<2.5:
                        color = "red" #
                        ct+=1
                        c0+=1
                    elif triang[1]<2.5:
                        color = "green"
                        ct+=1
                        c1+=1
                    elif triang[2]<2.5:
                        color = "blue"
                        ct+=1
                        c2+=1
                    elif triang[3]<2.5:
                        color = "orange"
                        ct+=1
                        c3+=1
                    else: color = "black"
                except: color = "black"
                draw.point([(j, i)], fill=color)
            i+=1
            
    elif OptShowB == "pixels": 
        if OptShowA == "numbers" and not TabGrains and OptShowA != "IPF":
            OptShowA = "nothing"
        while i < ny:
            for j in range(nx):
                try:
                    pix = TabPix[i * nx + j]
                    if pix[0]==0: pass
                    elif OptShowA == "euler1": color = int(((pix[0] - euler1min) / deuler1) * NbColors)
                    elif OptShowA == "euler2": color = int(((pix[1] - euler2min) / deuler2) * NbColors)
                    elif OptShowA == "euler3": color = int(((pix[2] - euler3min) / deuler3) * NbColors)
                    elif OptShowA == "euler123": color = int(((pix[0] + pix[1] + pix[2] - euler1min - euler2min - euler3min) /
                                                          (deuler1 + deuler2 + deuler3)) * NbColors)
                    elif OptShowA == "qual": color = int(((pix[5]-bcmin) / dbc) * NbColors) # grey scale
                    elif OptShowA == "IPF":
                        #CalcIPF(pix[0],pix[1],pix[2],invTSD,CrystD,[i * nx + j],IPFaxesD) je bloque ca car pour linstant trop lent
                        draw.point([(j, i)], fill=(255,255,255))
                    elif OptShowA == "nothing": color = 0
                    try:
                        if pix[0]==0: draw.point([(j, i)], fill=(255,255,255)) # (255,255,255) pour avoir le fond en blanc
                        elif OptShowA != "qual" and OptShowA != "IPF":
                            rgb = Palette[color]
                            f = (abs(pix[5]-bcmin) / dbc)
                            f=sqrt(f)
                            draw.point([(j, i)], fill=(int(f*rgb[0]),int(f*rgb[1]),int(f*rgb[2])))
                        elif OptShowA == "IPF": pass
                        else: draw.point([(j, i)], fill=(color,color,color))
                    except:
                        draw.point([(j, i)], fill=(255,255,255)) # (255,255,255) pour avoir le fond en blanc et (0,0,0) pour le noir
                except:
                    draw.point([(j, i)], fill=(0,0,0))
            i+=1
    
    elif OptShowB == "pixelsP": 
        while i < ny:
            for j in range(nx): 
                try:
                    pix = TabPixP[i * nx + j]
                    if pix[0]==0: pass
                    elif OptShowA == "euler1": color = int(((pix[0] - euler1minP) / deuler1P) * NbColors)
                    elif OptShowA == "euler2": color = int(((pix[1] - euler2minP) / deuler2P) * NbColors)
                    elif OptShowA == "euler3": color = int(((pix[2] - euler3minP) / deuler3P) * NbColors)
                    elif OptShowA == "euler123": color = int(((pix[0] + pix[1] + pix[2] - euler1minP - euler2minP - euler3minP) /
                                                          (deuler1P + deuler2P + deuler3P)) * NbColors)
                    elif OptShowA == "qual":
                        pix = TabPix[i * nx + j]
                        color = color = int(((pix[5]-bcmin) / dbc) * NbColors)
                    elif OptShowA == "IPF":
                        # CalcIPF(pix[0],pix[1],pix[2],invTSP,CrystP,[i * nx + j], IPFaxesP) # bloque car trop lent
                        draw.point([(j, i)], fill=(0,0,0))
                    elif OptShowA == "nothing": color = 0
                    try:
                        if pix[0]==0: draw.point([(j, i)], fill=(255,255,255))
                        elif OptShowA != "qual" and OptShowA != "IPF":
                            rgb = Palette[color]
                            draw.point([(j, i)], fill=(int(rgb[0]),int(rgb[1]),int(rgb[2])))
                        elif OptShowA == "IPF": pass
                        else: draw.point([(j, i)], fill=(color,color,color))
                    except:
                        draw.point([(j, i)], fill=(0,0,0))
                except: draw.point([(j, i)], fill=(0,0,0))
            i+=1

    elif OptShowB == "graind" and TabGrains and OptShowA != "IPF":
        while i < ny:
            for j in range(nx):
                try: indgr = TabPix[i * nx + j][3] - 1 #3 = graind
                except: indgr =-1
                if indgr != -1:
                    try:
                        gr = TabGrains[indgr]
                        if OptShowA == "euler1": color = int(((gr.euler1 - euler1min) / deuler1) * NbColors)
                        elif OptShowA == "euler2": color = int(((gr.euler2 - euler2min) / deuler2) * NbColors)
                        elif OptShowA == "euler3": color = int(((gr.euler3 - euler3min) / deuler3) * NbColors)
                        elif OptShowA == "euler123": color = int(((gr.euler1 + gr.euler2 + gr.euler3 - euler1min - euler3min - euler3min) /
                                                              (deuler1 + deuler2 + deuler3)) * NbColors)
                        elif OptShowA == "qual": color = int(((gr.confind - confmin) / dconf) * NbColors)
                        elif OptShowA == "numbers": color = int((float(RandList[indgr]) / lenTab) * NbColors)
                        elif OptShowA == "def2": color = int(((gr.kam-kammin) / kammax) * NbColors)

                    except : color = 0
                    try:
                        if OptShowA != "qual": draw.point([(j, i)], fill=Palette[color])
                        else: draw.point([(j, i)], fill=(color,color,color))
                    except: color = 0
                else: color = "white"
            i+=1

    elif OptShowB == "grainp" and OptShowA != "IPF":
        flagret = (len(TabRetainedGrains)>0)
        flagpar = (len(TabParentGrains)>0)
        while i < ny:
            for j in range(nx):
                try:
                    indgr = TabPix[i * nx + j][4] # 4=grainp reconstructed
                    ret = (TabPix[i * nx + j][5]==300)
                except:
                    #print "problem here in showmap", i, nx, j, len(TabPix), Nbpixx, Nbpixy
                    indgr, ret = 0,0
                gr = 0
                if indgr==0 or (flagret==0 and flagpar ==0):
                    draw.point([(j, i)], fill=(255, 255, 255))
                else:
                    if not ret and (OptShowRet ==0 or OptShowRet ==2):
                        indgr = indgr
                        gr = TabParentGrains[indgr-1]
                    #if ret and TabRetainedGrains and (OptShowRet == 1 or OptShowRet ==2):
                    if ret and (OptShowRet == 1 or OptShowRet ==2):
                        indgr = abs(TabPixP[i * nx + j][4])
                        gr = TabRetainedGrains[abs(indgr)-1]
                    if gr:
                        if OptShowA == "euler1": color = int(((gr.euler1 - euler1minP) / deuler1P) * NbColors)
                        elif OptShowA == "euler2": color = int(((gr.euler2 - euler2minP) / deuler2P) * NbColors)
                        elif OptShowA == "euler3": color = int(((gr.euler3 - euler3minP) / deuler3P) * NbColors)
                        elif OptShowA == "euler123": color = int(((gr.euler1 + gr.euler2 + gr.euler3 - euler1minP - euler3minP - euler3minP) /
                                                              (deuler1P + deuler2P + deuler3P)) * NbColors)
                        elif OptShowA == "qual": color = int(((gr.confind - confmin) / dconf) * NbColors)
                        elif OptShowA == "numbers":
                            if ret: color = int((float(RandList[abs(indgr)-1]) / lenTab) * NbColors)
                            else: color = int((float(RandList[indgr+len(TabRetainedGrains)-1]) / lenTab) * NbColors)
                        try:
                            if OptShowA != "qual": draw.point([(j, i)], fill=Palette[color])
                            else: draw.point([(j, i)], fill=(color,color,color))
                        except: pass
            i+=1

    elif (OptShowB == "graind" or OptShowB == "pixels")and TabGrains and OptShowA == "IPF":
        for grd in TabGrains: CalcIPF(grd.euler1,grd.euler2,grd.euler3,invTSD,CrystD,grd.listpix,IPFaxesD)
                
    elif (OptShowB == "grainp" or OptShowB == "pixelsP") and (TabParentGrains or TabRetainedGrains) and OptShowA == "IPF":
        if TabParentGrains and (OptShowRet ==0 or OptShowRet ==2):
            for grp in TabParentGrains: CalcIPF(grp.euler1,grp.euler2,grp.euler3,invTSP,CrystP,grp.listpix,IPFaxesP)      
        if TabRetainedGrains and (OptShowRet ==1 or OptShowRet ==2):
            for grp in TabRetainedGrains: CalcIPF(grp.euler1,grp.euler2,grp.euler3,invTSP,CrystP,grp.listpix,IPFaxesP)
        
    dLmin = nx
    L = nx
    listL = [0.01,0.05,0.1,0.2,0.5,1,2,5,10,20,30,40,50,100,200,300,400,500,1000,2000,3000,4000,5000,10000]
    for Li in listL:
        ix = int(Li/XStep)
        dL = abs(nx/3-ix)
        if dL<dLmin:
            L = Li
            dLmin = dL
    ScaleBar = int(L / XStep)
    
    canvas["height"] = min(int(ny * Scale + 40 * Scale), HMAX - 200)
    canvas["width"] = min(int(nx * Scale), WMAX - 500)
    scanH = max(int(ny * Scale), 1.5 * int(ny * Scale))
    scanW = max(int(nx * Scale), 1.5 * int(nx * Scale))
    canvas["scrollregion"] = (0, 0, scanW, scanH)

    if GRID != "Hexagon": canvas.create_line(int(3 * Scale), int((ny + 5) * Scale), int((ScaleBar + 3) * Scale), int((ny + 5) * Scale), width=max(int(Scale), 2), fill="Black")
    else: canvas.create_line(int(3 * Scale), int((ny + 5) * Scale*(sqrt(3) / 2)), int((ScaleBar + 3) * Scale), int((ny + 5) * Scale*(sqrt(3) / 2)), width=max(int(Scale), 2), fill="Black")
    nny = ny
    if Scale < 1: nny = nny + int(5. / Scale)

    try :
        sb = canvas.find_withtag("scalebar")
        for wd in sb: canvas.delete(wd)
    except: pass

    if GRID != "Hexagon":
        textsb = canvas.create_text(int((3+ScaleBar/2) * Scale), int((nny + 25) * Scale), text=str(L) + " " + Units, fill="black", anchor="center", font="arial " + str(int(SBSize * Scale)))
    else:
        textsb = canvas.create_text(int((3+ScaleBar/2) * Scale), int((nny + 25) * Scale *(sqrt(3) / 2)), text=str(L) + " " + Units, fill="black", anchor="center", font="arial " + str(int(SBSize * Scale)))

    canvas.itemconfig(textsb, tags="scalebar")
    
    if GRID != "Hexagon": image = image.resize((int(nx * Scale), int(ny * Scale)))
    else: image = image.resize((int(nx * Scale), int(ny * Scale * sqrt(3) / 2)))

    imageTk = ImageTk.PhotoImage(image)
    if GRID != "Hexagon": imageCV = canvas.create_image(int(nx * Scale / 2), int(ny * Scale / 2), image=imageTk)
    else: imageCV = canvas.create_image(int(nx * Scale / 2), int(ny *(sqrt(3) / 2)* Scale / 2), image=imageTk)
    canvas.itemconfig(imageCV, tags=("image"))

    if DaughterGB.get()and TabGrains: showGBD() #fera GBP au besoin, pour etre sur de ecrire les GBP au dessus des GBD
    elif ParentGB.get()and TabParentGrains: showGBP()
    else:
        if OptShowN: showGrainNum() # affiche les numeros des grains si pas affiches voa showGBD or GBP

    if OptShowA == "qual" and OptShowB == "pixels":
        EtatRun.set("\n"+"Image of Band Contrast"+"\n")
        LabelRun["fg"] = "blue"
    elif OptShowA == "def" or OptShowA == "def2":
        t = "KAM on scale from "+ cut(defmin)+" to " + cut(defmax)+" Deg. Mean KAM = " + cut(KAMmean)
        EtatRun.set("\n"+t+"\n")
        LabelRun["fg"] = "blue"
    elif OptShowA == "gos":
        t = "GOS on scale from "+ cut(gosmin)+" to " + cut(gosmax)+" Deg. Mean GOS = " + cut(GOSmean)
        EtatRun.set("\n"+t+"\n")
        LabelRun["fg"] = "blue"  
    elif OptShowA == "Triangle":
        pass
    elif OptShowA == "Triangle2":
        c0 = float(c0*100)/ct
        c1 = float(c1*100)/ct
        c2 = float(c2*100)/ct
        c3 = float(c3*100)/ct
        t = "Surf. Proportion : Pitsch = "+ cut(c0) + "%" + ", KS =" + cut(c1)+ "%"+ ", NW =" + cut(c2)+"%" + ", and GT = " + cut(c3)+"%"
        wd = "P=Red, KS=Green, NW=Blue, and G=Orange within 2.5 deg. tolerance"
        EtatRun.set(wd+".\n"+ResTri+"\n"+t)
        LabelRun["fg"] = "blue"
    elif OptShowA == "CPDPackets":
        t = "Close-Packed Direction Packets (6 colours)"
        EtatRun.set("\n"+t+"\n")
        LabelRun["fg"] = "blue"
    elif OptShowA == "CPPPackets":
        t = "Close-Packed Plane Packets (Morito, 4 colours)"
        EtatRun.set("\n"+t+"\n")
        LabelRun["fg"] = "blue"
    elif OptShowA == "BainPackets":
        t = "Bain Packets (Low-Misorientated variants, 3 colours)"
        EtatRun.set("\n"+t+"\n")
        LabelRun["fg"] = "blue"
    elif OptShowA == "BurstPackets":
        t = "Plate Packets (ZigZag or Branched variants, 6 colours)"
        EtatRun.set("\n"+t+"\n")
        LabelRun["fg"] = "blue" 
    elif OptShowA == "qual" and OptShowB == "graind":
        EtatRun.set("\n"+"Image of Quality Index after Parent Reconstruction"+"\n")
        LabelRun["fg"] = "blue"
    elif OptShowA == "qual" and OptShowB == "grainp":
        EtatRun.set("\n"+"Image of Mean Quality Index after Parent Reconstruction"+"\n")
        LabelRun["fg"] = "blue"
    elif OptShowB == "pixels" and OptShowA != "IPF":
        EtatRun.set("\n"+"Map of Daughter Pixels"+"\n")
        LabelRun["fg"] = "blue"
    elif OptShowB == "pixelsP" and OptShowA != "IPF":
        EtatRun.set("\n"+"Map of Parent Pixels"+"\n")
        LabelRun["fg"] = "blue"
    elif OptShowA == "IPF":
        if modeIPF==2: wd = "\n"+infoRet+" IPF_Z "+"\n"
        elif modeIPF==1: wd = "\n"+infoRet+" IPF_Y "+"\n"
        elif modeIPF==0: wd = "\n"+infoRet+" IPF_X "+"\n"
        EtatRun.set(wd)
        LabelRun["fg"] = "blue"
    else:
        pass
        #EtatRun.set("\n"+"Map ready."+"\n")
        #LabelRun["fg"] = "blue"

#===================================================================================================
    
def ShowPackets100 (event):
    
    global OptShowA
    refreshButtons2()
    ButtonM["foreground"] = "white"
    ButtonM["background"] = "blue"
    
    OptShowA = "packets100"
    showmap()
#===================================================================================================
    
def ShowPackets010 (event):
    
    global OptShowA
    refreshButtons2()
    ButtonM["foreground"] = "white"
    ButtonM["background"] = "blue"
    
    OptShowA = "packets010"
    showmap()
    #===================================================================================================
    
def ShowPackets001 (event):
    
    global OptShowA
    refreshButtons2()
    ButtonM["foreground"] = "white"
    ButtonM["background"] = "blue"
    
    OptShowA = "packets001"
    showmap()
  
#===================================================================================================
    
def ShowCPPPackets (event):
    global Transition,TransLoadMode
    global OptShowA
    
    if Transition != "Fe-KS":
        info1 = "The packets are calculated from KS variants.\n The CPP packets are the variants sharing the same (110)bcc // (111)fcc plane. \n"
        info2 = "The variants must be re-indexed with KS OR. \n To come back to yout initial OR if it is not KS: reload your OR and choose Absolute Variant Indexing"
        tkMessageBox.showinfo("Help", info1 + info2)
        Transition = "Fe-KS"
        TransLoadMode = 1
        LoadTransitionFiles(0)
        AbsoluteVarNum()
        
    refreshButtons2()
    ButtonM["foreground"] = "white"
    ButtonM["background"] = "blue"
    OptShowA = "CPPPackets"
    showmap()

#===================================================================================================
    
def ShowCPDPackets (event):    
    global Transition,TransLoadMode
    global OptShowA
    
    if Transition != "Fe-KS":
        info1 = "The packets are calculated from KS variants.\n The CPD packets are the variants sharing the same [111]bcc // [110] fcc direction. \n"
        info2 = "The variants must be re-indexed with KS OR. \n To come back to yout initial OR if it is not KS: reload your OR and choose Absolute Variant Indexing"
        tkMessageBox.showinfo("Help", info1 + info2)
        Transition = "Fe-KS"
        TransLoadMode = 1
        LoadTransitionFiles(0)
        AbsoluteVarNum()
        
    refreshButtons2()
    ButtonM["foreground"] = "white"
    ButtonM["background"] = "blue"
    OptShowA = "CPDPackets"
    showmap()
#===================================================================================================
    
def ShowBainZones (event):
    global Transition, TransLoadMode
    global OptShowA

    if Transition != "Fe-KS":
        info1 = "The packets are calculated from KS variants.\n The Bain packets are the variants linked by low-misorientations (operators 2A and 2B) \n"
        info2 = "The variants must be re-indexed with KS OR. \n To come back to yout initial OR if it is not KS: reload your OR and choose Absolute Variant Indexing"
        tkMessageBox.showinfo("Help", info1 + info2)
        Transition = "Fe-KS"
        TransLoadMode = 1
        LoadTransitionFiles(0)
        AbsoluteVarNum()

    refreshButtons2()
    ButtonM["foreground"] = "white"
    ButtonM["background"] = "blue"
    OptShowA = "BainPackets"
    showmap()
#===================================================================================================
    
def ShowBurstZone (event):
    global Transition, TransLoadMode
    global OptShowA

    if Transition != "Fe-KS":
        info1 = "The packets are calculated from KS variants.\n The plate packets are the variants in zig-zag and branched martensite (High carbon steels)\n"
        info2 = "The variants must be re-indexed with KS OR. \n To come back to yout initial OR if it is not KS: reload your OR and choose Absolute Variant Indexing"
        tkMessageBox.showinfo("Help", info1 + info2)
        Transition = "Fe-KS"
        TransLoadMode = 1
        LoadTransitionFiles(0)
        AbsoluteVarNum()

    refreshButtons2()
    ButtonM["foreground"] = "white"
    ButtonM["background"] = "blue"
    OptShowA = "BurstPackets"
    showmap()
#===================================================================================================
def ShowCPPPackets0():
    ShowCPPPackets(0)
def ShowCPDPackets0():
    ShowCPDPackets(0)
def ShowBainZones0():
    ShowBainZones (0)
def ShowBurstZone0():
    ShowBurstZone (0)
#===================================================================================================

def hideGrainNum():
    
    global canvas, imageTk, imageCV, image

    all_text = canvas.find_withtag("GNum")
    for wd in all_text: canvas.delete(wd)
    all_text = canvas.find_withtag("GNum2")
    for wd in all_text: canvas.delete(wd)

#===================================================================================================   

def showGrainNum():
    
    global canvas, imageTk, imageCV
    
    nx = Nbpixx
    ny = Nbpixy

    print Xmax, Ymax, nx, ny, Scale

    if OptShowB == "graind" and TabGrains:
        font = "Arial "+ str(NumSizeSmall)
        f = float(NumSizeSmall)/10
        for gr in TabGrains:
            wd = str(gr.graind)
            if GRID != "Hexagon": x, y = int((gr.x / Xmax) * nx * Scale), int((gr.y / Ymax) * ny * Scale)
            else: x, y = int((gr.x / Xmax) * nx * Scale), int((gr.y / Ymax) * ny * Scale * sqrt(3)/2) 
            fondtext = canvas.create_rectangle(x - int(3*len(wd)*f), y - int(6*f), x + int(3*len(wd)*f), y + int(6*f), fill="white", outline="white")
            text = canvas.create_text(x, y, text=wd, fill="black", font = font)
            canvas.itemconfig(text, tags="GNum")
            canvas.itemconfig(fondtext, tags="GNum")

    elif OptShowB == "grainp" and TabParentGrains:
        font = "Arial "+ str(NumSizeBig)
        f = float(NumSizeBig)/20
        for gr in TabParentGrains:
            wd = str(gr.grainp)
            if GRID != "Hexagon": xt, yt = int((gr.x / Xmax) * nx * Scale), int((gr.y / Ymax) * ny * Scale)
            else: xt, yt = int((gr.x / Xmax) * nx * Scale), int((gr.y / Ymax) * ny * Scale * sqrt(3)/2)
            fondtext = canvas.create_rectangle(xt - int(10*len(wd)*f), yt - int(15*f), xt + int(10*len(wd)*f), yt + int(15*f), fill="white", outline="white")
            text = canvas.create_text(xt, yt, text=wd, fill="black", font = font)
            canvas.itemconfig(text, tags="GNum")
            canvas.itemconfig(fondtext, tags="GNum")
    else: pass

#===================================================================================================   

def hideGBD():
    
    global canvas, imageTk, imageCV

    all_gb = canvas.find_withtag("GBD")
    for gb in all_gb: canvas.lower(gb) #lift / lower

#===================================================================================================

def writeGBD():
    
    global canvas, imageTk, imageCV, image

    all_gb = canvas.find_withtag("GBD")
    for gb in all_gb: canvas.delete(gb)

    if NextStep > 4:
        nx = Nbpixx
        ny = Nbpixy
        di = 1
        dj = 1

        if TabGrains:
            for gr in TabGrains:
                gbd = []
                for pix in gr.listpixfr:
                    i = pix / nx
                    j = pix - nx * i
                    if GRID != "Hexagon": gbd = canvas.create_rectangle(int(j * Scale), int(i * Scale), int((j + dj) * Scale), int((i + di) * Scale), fill="grey20", outline="")
                    else:
                        gbd = canvas.create_rectangle(int(j * Scale), int(i * Scale * sqrt(3) / 2),
                                    int((j + dj) * Scale), int((i + di) * Scale * sqrt(3) / 2), fill="grey20", outline="")
                    #canvas.itemconfig(gbd, tags="GBD",state=HIDDEN)
                    canvas.itemconfig(gbd, tags="GBD")

#===================================================================================================

def showGBD():
    
    global canvas, imageTk, imageCV, image

    if DaughterGB.get():

        all_gb = canvas.find_withtag("GBD")
        for gb in all_gb: canvas.lift(gb) #lift / lower

        if len(all_gb) == 0: writeGBD()
        if ParentGB.get() and TabParentGrains: showGBP()

    else: hideGBD()

    if OptShowN: showGrainNum()

#===================================================================================================

def hideGBP():
    
    global canvas, imageTk, imageCV, image

    all_gb = canvas.find_withtag("GBP")
    for wd in all_gb: canvas.lower(wd)
    all_gb = canvas.find_withtag("GBPS")
    for wd in all_gb: canvas.lower(wd)

#===================================================================================================   

def writeGBP():
    
    global canvas, imageTk, imageCV, image

    all_gb = canvas.find_withtag("GBP")
    for gb in all_gb: canvas.delete(gb)
    all_gb = canvas.find_withtag("GBPS")
    for gb in all_gb: canvas.delete(gb)

    nx = Nbpixx
    ny = Nbpixy

    di = 2
    dj = 2

    if TabParentGrains:
        for gr in TabParentGrains:
            for pix in gr.listpixfr:
                i = pix / nx
                j = pix - nx * i
                if GRID != "Hexagon": gbp = canvas.create_rectangle(int(j * Scale), int(i * Scale), int((j + dj) * Scale), int((i + di) * Scale), fill="Black", outline="")
                else: gbp = canvas.create_rectangle(int(j * Scale), int(i * Scale * sqrt(3) / 2),
                                int((j + dj) * Scale), int((i + di) * Scale * sqrt(3) / 2), fill="Black", outline="")
                canvas.itemconfig(gbp, tags="GBP")

    if TabRetainedGrains:
        for gr in TabRetainedGrains:
            for pix in gr.listpixfr:
                i = pix / nx
                j = pix - nx * i
                if GRID != "Hexagon": gbp = canvas.create_rectangle(int(j * Scale), int(i * Scale), int((j + dj) * Scale), int((i + di) * Scale), fill="Black", outline="")
                else: gbp = canvas.create_rectangle(int(j * Scale), int(i * Scale * sqrt(3) / 2),
                                int((j + dj) * Scale), int((i + di) * Scale * sqrt(3) / 2), fill="Black", outline="")
                canvas.itemconfig(gbp, tags="GBP")

#===================================================================================================   

def showGBP():
    
    global canvas, imageTk, imageCV, image

    if ParentGB.get():

        all_gb = canvas.find_withtag("GBP")
        for gb in all_gb: canvas.lift(gb) #lift / lower

        if len(all_gb) == 0:
            writeGBP()

    else: hideGBP()

    if OptShowN: showGrainNum()

#===================================================================================================   

def showGrey():
    
    global canvas, imageTk, imageCV, image
    
    nx = Nbpixx
    ny = Nbpixy

    if GreyFlag.get():
        greyrect = canvas.create_rectangle(0, 0, nx * Scale, ny * Scale, fill="dark grey", outline="")
        canvas.itemconfig(greyrect, tags="grey")

        try:
            all_gb = canvas.find_withtag("GBD")
            for gb in all_gb: canvas.lift(gb) #lift / lower
        except: pass
        try:
            all_gb = canvas.find_withtag("GBP")
            for gb in all_gb: canvas.lift(gb) #lift / lower
        except: pass
        if OptShowN: showGrainNum()

            #if DaughterGB.get() and TabGrains: DrawGBSigma3n()

    else:
        greyrect = canvas.find_withtag("grey")
        for wd in greyrect: canvas.delete(wd)

 #===================================================================================================   

def hideMap():
    
    global canvas, imageTk, imageCV, image

    if HideFlag.get():
        canvas.delete(ALL)
        canvas["height"] = 0
        canvas["width"] = 0

    else: showmap()

#===================================================================================================   

def showVarNum():
    
    global canvas, imageTk, imageCV, image
    
    nx = Nbpixx
    ny = Nbpixy

    if not VarNum.get():
        all_text = canvas.find_withtag("VarNum")
        for wd in all_text: canvas.delete(wd)
        all_text = canvas.find_withtag("VarNum2")
        for wd in all_text: canvas.delete(wd)
    else:
        if TabParentGrains:
            for grp in TabParentGrains:
                for indgr in grp.graind:
                    gr = TabGrains[indgr - 1]
                    if GRID != "Hexagon": x, y = int((gr.x / Xmax) * nx * Scale), int((gr.y / Ymax) * ny * Scale)
                    else: x, y = int((gr.x / Xmax) * nx * Scale), int((gr.y / Ymax) * ny * Scale*sqrt(3)/2)
                    if Transition != "Sigma3":
                        if gr.variant: #n affiche que si un seul numero est trouve
                            wd = str(gr.variant[0])
                            fondtext = canvas.create_rectangle(x - 6, y - 6, x + 6, y + 6, fill="white", outline="white", tags="VarNum2")
                            text = canvas.create_text(x, y, text=wd, fill="black", font="Times 12 bold", tags="VarNum")
                    else:
                            try: wd = str(listTwinNom[gr.variant[0]][0]) + " , '" + stringify(gr.variant[1]) + "'"
                            except : wd = ''
                            #wdt = wd[1:-1] #enleve les [ ]
                            l = len(wd)
                            fondtext = canvas.create_rectangle(x - 3 * l, y - 6, x + 3 * l, y + 6, fill="white", outline="white", tags="VarNum2")
                            text = canvas.create_text(x, y, text=wd, fill="black", font="Times 12 bold", tags="VarNum")


#===================================================================================================
    
def calculateHabitPlanes ():
    global canvas, imageTk, imageCV, image, tolHP
    
    nx = Nbpixx
    ny = Nbpixy
    tolHP = 5.0
    listHPs = []
    fHP = Toplevel(root)
    t = "______________________________________ Habit Planes ___________________________________"
    lt = Label(fHP, text=t, foreground="black",font = ("Helvetica", "12", "underline"))
    lt.grid(row=2, column=2, columnspan = 10,ipady=5, sticky=E+W)

    fadd = Frame(fHP)
    fadd.grid(row=4, column=2, ipadx=2, sticky=W)
    fhkl = Frame(fadd)
    fhkl.grid(row=1, column=1, sticky=E + W)
    lt = Label(fhkl, text=" Daughter HP: {", foreground="black").grid(row=1, column=1, sticky=E)
    lt = Label(fhkl, text="}", foreground="black").grid(row=1, column=5, sticky=W)
    lt = Label(fhkl, text=" Parent HP: {", foreground="black").grid(row=1, column=11, sticky=E)
    lt = Label(fhkl, text="}", foreground="black").grid(row=1, column=15, sticky=W)
    lt = Label(fhkl, text=" Tolerance angle:", foreground="black").grid(row=1, column=20, sticky=W)
    entry_tolHP = Entry(fhkl, width=3, foreground="blue")
    entry_tolHP.insert(0, tolHP)
    entry_tolHP.grid(row=1, column=25, sticky=E + W)
    lt = Label(fhkl, text=" Deg.", foreground="black").grid(row=1, column=30, sticky=W)
    

    hkl=[1,1,0]
    entry_paramHPd = [None for i in range(3)]
    for j in range(3):
        entry_paramHPd[j] = Entry(fhkl, width=3, foreground="blue")
        entry_paramHPd[j].insert(0, hkl[j])
        entry_paramHPd[j].grid(row=1, column=j + 2, sticky=E + W)

    hkl=[1,1,1]
    entry_paramHPp = [None for i in range(3)]
    for j in range(3):
        entry_paramHPp[j] = Entry(fhkl, width=3, foreground="blue")
        entry_paramHPp[j].insert(0, hkl[j])
        entry_paramHPp[j].grid(row=1, column=j + 12, sticky=E + W)

    fmet = Frame(fadd)
    fmet.grid(row=2, column=1, sticky=E + W)
    lt = Label(fmet, text="", foreground="black").grid(row=4, column=1, sticky=E + W) 
    lt = Label(fmet, text="Choose your method:", foreground="black",font = ("Helvetica", "12", "underline")).grid(row=5, column=1,ipady=2, sticky=E + W)   
    Radiobutton (fmet, text="Only Daughter HPs", variable=HPPhaseMethod, value=0).grid (row=6, column=1, sticky=W)
    Radiobutton (fmet, text="Only Parent HPs ", variable=HPPhaseMethod, value=1).grid (row=6, column=2, sticky=W)
    Radiobutton (fmet, text="Daughter & Parent HPs ", variable=HPPhaseMethod, value=2).grid (row=6, column=3, sticky=W)
    Radiobutton (fmet, text="Only Daughter HPs // Parent HPs ", variable=HPPhaseMethod, value=3).grid (row=6, column=4, sticky=W)

    fcol = Frame(fHP)
    fcol.grid(row=6, column=2, ipadx=1, sticky=E + W)
    lt = Label(fcol, text="", foreground="black").grid(row=1, column=1, sticky=E + W)
    lt = Label(fcol, text="Daughter HP", foreground="black").grid(row=2, column=1, sticky=E + W)
    lt = Label(fcol, text=" Color", foreground="black").grid(row=2, column=3, sticky=E + W)
    Radiobutton (fcol, text="Black", variable=HPdColorMethod, value=0).grid (row=2, column=4, sticky=W)
    Radiobutton (fcol, text="White", variable=HPdColorMethod, value=1).grid (row=2, column=5, sticky=W)
    Radiobutton (fcol, text="Grey", variable=HPdColorMethod, value=2).grid (row=2, column=6, sticky=W)
    Radiobutton (fcol, text="Blue", variable=HPdColorMethod, value=3).grid (row=2, column=7, sticky=W)
    Radiobutton (fcol, text="Red", variable=HPdColorMethod, value=4).grid (row=2, column=8, sticky=W)
    Radiobutton (fcol, text="Green", variable=HPdColorMethod, value=5).grid (row=2, column=9, sticky=W)
    lt = Label(fcol, text="     Width", foreground="black").grid(row=2, column=10, sticky=E + W)
    entry_widthHPd = Entry(fcol, width=3, foreground="blue")
    entry_widthHPd.insert(0, 1)
    entry_widthHPd.grid(row=2, column=11, sticky=E + W)

    fcol2 = Frame(fHP)
    fcol2.grid(row=7, column=2, ipadx=1, sticky=E + W)
    lt = Label(fcol2, text="", foreground="black").grid(row=1, column=1, sticky=E + W)
    lt = Label(fcol2, text="Parent HP    ", foreground="black").grid(row=2, column=1, sticky=E + W)
    lt = Label(fcol2, text=" Color", foreground="black").grid(row=2, column=3, sticky=E + W)
    Radiobutton (fcol2, text="Black", variable=HPpColorMethod, value=0).grid (row=2, column=4, sticky=W)
    Radiobutton (fcol2, text="White", variable=HPpColorMethod, value=1).grid (row=2, column=5, sticky=W)
    Radiobutton (fcol2, text="Grey", variable=HPpColorMethod, value=2).grid (row=2, column=6, sticky=W)
    Radiobutton (fcol2, text="Blue", variable=HPpColorMethod, value=3).grid (row=2, column=7, sticky=W)
    Radiobutton (fcol2, text="Red", variable=HPpColorMethod, value=4).grid (row=2, column=8, sticky=W)
    Radiobutton (fcol2, text="Green", variable=HPpColorMethod, value=5).grid (row=2, column=9, sticky=W)
    lt = Label(fcol2, text="     Width", foreground="black").grid(row=2, column=10, sticky=E + W)
    entry_widthHPp = Entry(fcol2, width=3, foreground="blue")
    entry_widthHPp.insert(0, 1)
    entry_widthHPp.grid(row=2, column=11, sticky=E + W)
    lt = Label(fcol2, text="(0 for invisibility)", foreground="black").grid(row=2, column=12, sticky=E + W)
    lt = Label(fcol2, text="", foreground="black").grid(row=5, column=1, sticky=E + W)

    fmode = Frame(fHP)
    fmode.grid(row=8, column=2, ipadx=2, sticky=E + W)
    lt = Label(fmode, text="Action: ", foreground="black", font = ("Helvetica", "12", "underline")).grid(row=2, column=1, sticky=E + W)
    #Radiobutton (fmode, text="None", variable=HPModeMethod, value=0).grid (row=2, column=2, sticky=W)
    Radiobutton (fmode, text=" Click to Select & Click to Place", variable=HPModeMethod, value=1).grid (row=3, column=2, sticky=W)
    Radiobutton (fmode, text="Click to Select & Click to Duplicate", variable=HPModeMethod, value=2).grid (row=4, column=2, sticky=W)
    Radiobutton (fmode, text=" Click to Select & Press +/- to scale up or r/g/b/y/p to change the color, and <Return> to complete", variable=HPModeMethod, value=3).grid (row=5, column=2, sticky=W)
    Radiobutton (fmode, text="Click to Select & Click to Delete", variable=HPModeMethod, value=4).grid (row=6, column=2, sticky=W)
    Radiobutton (fmode, text="Click to Select & Click to Write the Plane Label in the Image", variable=HPModeMethod, value=5).grid (row=7, column=2, sticky=W)


    def ShowHPs():
        global canvas, imageTk, imageCV,image, listHPs, tolHP

        all_hp = canvas.find_withtag("HP")
        for hp in all_hp: canvas.delete(hp)

        all_hp = canvas.find_withtag("wHP")
        for hp in all_hp: canvas.delete(hp)

        
        def attrD():
            TabG = TabGrains
            Group = GroupeRotD
            TM = TMD
            TS = TSD
            h = string.atoi(entry_paramHPd[0].get())
            k = string.atoi(entry_paramHPd[1].get())
            l = string.atoi(entry_paramHPd[2].get())
            width = string.atoi(entry_widthHPd.get())
            col = "black"
            ncol = HPdColorMethod.get()
            if ncol == 0: col = "black"
            elif ncol ==1: col = "white"
            elif ncol ==2: col = "grey"
            elif ncol ==3: col = "blue"
            elif ncol ==4: col = "red"
            elif ncol ==5: col = "green"
            HP = array([h, k, l])
            return HP,TabG,Group,TM,TS,col,width
        
        def attrP():
            TabG = TabParentGrains
            Group = GroupeRotP
            TM = TMP
            TS = TSP
            h = string.atoi(entry_paramHPp[0].get())
            k = string.atoi(entry_paramHPp[1].get())
            l = string.atoi(entry_paramHPp[2].get())
            width = string.atoi(entry_widthHPp.get())
            col = "black"
            ncol = HPpColorMethod.get()
            if ncol == 0: col = "black"
            elif ncol ==1: col = "white"
            elif ncol ==2: col = "grey"
            elif ncol ==3: col = "blue"
            elif ncol ==4: col = "red"
            elif ncol ==5: col = "green"
            HP = array([h, k, l])
            return HP,TabG,Group,TM,TS,col,width

        method = HPPhaseMethod.get()
        if method==0:
            infoHP = attrD()
            PlotHPs(infoHP)
        elif method==1:
            infoHP = attrP()
            PlotHPs(infoHP)
        elif method==2:
            infoHP = attrD()
            PlotHPs(infoHP)
            infoHP = attrP()
            PlotHPs(infoHP)
        elif method==3:
            infoHPd = attrD()
            infoHPp = attrP()
            tolHP = string.atof(entry_tolHP.get())
            if string.atoi(entry_widthHPp.get()): PlotHPs(infoHPp)
            PlotParallelHPs(infoHPd,infoHPp)
        

    def PlotHPs(infoHP):
        global canvas, imageTk, imageCV,image, listHPs
        
        HP,TabG,Group,TM,TS,col,width = infoHP
        listHPs=[]
        listHPs.append(HP)
        for gi in Group:
            HPi = dot(transpose(gi),HP)
            flag = 1
            for HPj in listHPs:
                if HPi.tolist()== HPj.tolist() or HPi.tolist()== (-HPj).tolist(): flag = 0
            if flag : listHPs.append(HPi)
        for gr in TabG:
            w = 0.5*Scale*sqrt(len(gr.listpix))
            if GRID != "Hexagon": xt, yt = int((gr.x / Xmax) * nx * Scale), int((gr.y / Ymax) * ny * Scale)
            else: xt, yt = int((gr.x / Xmax) * nx * Scale), int((gr.y / Ymax) * ny * Scale * sqrt(3)/2)
            m_CtoS = EulersToMat(gr.euler1, gr.euler2, gr.euler3,SourceFile) #transf matrix crystal ref cubique to sample
            m_CtoS = dot(inv(TS), m_CtoS) #transf matrix crystal to sample
            normsurfC = dot(m_CtoS,array([0,0,1]))
            gsurfC = dot(TM, normsurfC)
            for HPi in listHPs:
                lineC = cross(gsurfC,HPi)
                lineS = dot(inv(m_CtoS),lineC)
                xl,yl = lineS[0],lineS[1]
                jl,il = w*normalize(array([xl,yl]))
                if GRID != "Hexagon": i1,j1,i2,j2 = xt - il, yt - jl, xt + il, yt + jl
                else: i1,j1,i2,j2 = xt - xl, yt - yl, xt + xl, yt + yl
                HPTrace = canvas.create_line(i1,j1,i2,j2, fill=col, width = width)
                canvas.itemconfig(HPTrace, tags=("HP",str(HPi)))
       

    def PlotParallelHPs(infoHPd,infoHPp):
        global canvas, imageTk, imageCV,image
        
        HPd,TabGd,Groupd,TMd,TSd,cold,widthd = infoHPd
        HPp,TabGp,Groupp,TMp,TSp,colp,widthp = infoHPp
        listHPd=[HPd]
        listHPp=[HPp]
        for gi in Groupd:
            HPi = dot(transpose(gi),HPd)
            flag = 1
            for HPj in listHPd:
                if HPi.tolist()== HPj.tolist() or HPi.tolist()== (-HPj).tolist(): flag = 0
            if flag : listHPd.append(HPi)
        for gi in Groupp:
            HPi = dot(transpose(gi),HPp)
            flag = 1
            for HPj in listHPp:
                if HPi.tolist()== HPj.tolist() or HPi.tolist()== (-HPj).tolist(): flag = 0
            if flag : listHPp.append(HPi)
        for grp in TabGp:
            PtoS = EulersToMat(grp.euler1, grp.euler2, grp.euler3,SourceFile) #transf matrix crystal ref cubique to sample
            PtoS = dot(inv(TSp), PtoS) #transf matrix crystal to sample
            StoP = inv(PtoS)
            listHPpinS = [dot(StoP,hp) for hp in listHPp]
            for indgrd in grp.graind:
                grd = TabGrains[indgrd - 1]
                w = 0.5*Scale*sqrt(len(grd.listpix))
                if GRID != "Hexagon": xt, yt = int((grd.x / Xmax) * nx * Scale), int((grd.y / Ymax) * ny * Scale)
                else: xt, yt = int((grd.x / Xmax) * nx * Scale), int((grd.y / Ymax) * ny * Scale * sqrt(3)/2)
                DtoS = EulersToMat(grd.euler1, grd.euler2, grd.euler3,SourceFile) #S sample, grd daughter grains
                DtoS = dot(invTSD, DtoS)
                StoD = inv(DtoS)
                listHPdinS = [dot(StoD,hp) for hp in listHPd]
                listHPdParall = []
                i=0
                for HPiS in listHPdinS:
                    for hppS in listHPpinS:
                        if abs(angle(HPiS,hppS))< tolHP or abs(abs(angle(HPiS,hppS))-180)< tolHP: # angular tolerance = 5 deg by default
                            lineS = cross(array([0,0,1]),HPiS)
                            xl,yl = lineS[0],lineS[1]
                            jl,il = w*normalize(array([xl,yl]))
                            if GRID != "Hexagon": i1,j1,i2,j2 = xt - il, yt - jl, xt + il, yt + jl
                            else: i1,j1,i2,j2 = xt - xl, yt - yl, xt + xl, yt + yl
                            HPTrace = canvas.create_line(i1,j1,i2,j2, fill=cold, width = widthd)
                            canvas.itemconfig(HPTrace, tags=("HP",str(listHPd[i])))
                            break
                    i+=1
                            

    lt = Label(fHP, text="", foreground="black").grid(row=9, column=2, sticky=E + W)
    b = Button(fHP, text="Show Habit Planes", command=ShowHPs)
    b.grid(row=10, column=2, sticky=E + W)
    b["foreground"] = "blue"
    b["background"] = "grey"
    lt = Label(fHP, text="", foreground="black").grid(row=11, column=2, sticky=E + W)

#===================================================================================================
    
def calculateJunctionPlanes ():
    global canvas, imageTk, imageCV, image, tolHP, nbOp
    
    nx = Nbpixx
    ny = Nbpixy
    tolHP = 5.0
    listHPs = []
    nbOp = [1]
    fHP = Toplevel(root)
    t = "______________________________________ Junction Planes ___________________________________"
    lt = Label(fHP, text=t, foreground="black",font = ("Helvetica", "12", "underline"))
    lt.grid(row=2, column=2, columnspan = 10,ipady=5, sticky=E+W)

    fadd = Frame(fHP)
    fadd.grid(row=4, column=2, ipadx=2, sticky=W)
    fhkl = Frame(fadd)
    fhkl.grid(row=1, column=1, sticky=E + W)
    lt = Label(fhkl, text=" Daughter 1 junction plane: {", foreground="black").grid(row=1, column=1, sticky=E)
    lt = Label(fhkl, text="}", foreground="black").grid(row=1, column=5, sticky=W)
    lt = Label(fhkl, text=" Daughter 2 junction plane: {", foreground="black").grid(row=1, column=11, sticky=E)
    lt = Label(fhkl, text="}", foreground="black").grid(row=1, column=15, sticky=W)
    lt = Label(fhkl, text=" Tolerance angle:", foreground="black").grid(row=1, column=20, sticky=W)
    entry_tolHP = Entry(fhkl, width=3, foreground="blue")
    entry_tolHP.insert(0, tolHP)
    entry_tolHP.grid(row=1, column=25, sticky=E + W)
    lt = Label(fhkl, text=" Deg.", foreground="black").grid(row=1, column=30, sticky=W)

    lt = Label(fhkl, text=" Operator(s) (ex: 1 or 1,2):", foreground="black").grid(row=1, column=40, sticky=W)
    entry_nbop = Entry(fhkl, width=10, foreground="blue")
    entry_nbop.insert(0, nbOp[0])
    entry_nbop.grid(row=1, column=45, sticky=E + W)
    

    hkl=[1,1,1]
    entry_paramHPd = [None for i in range(3)]
    for j in range(3):
        entry_paramHPd[j] = Entry(fhkl, width=3, foreground="blue")
        entry_paramHPd[j].insert(0, hkl[j])
        entry_paramHPd[j].grid(row=1, column=j + 2, sticky=E + W)

    hkl=[1,1,1]
    entry_paramHPp = [None for i in range(3)]
    for j in range(3):
        entry_paramHPp[j] = Entry(fhkl, width=3, foreground="blue")
        entry_paramHPp[j].insert(0, hkl[j])
        entry_paramHPp[j].grid(row=1, column=j + 12, sticky=E + W)

    fmet = Frame(fadd)
    fmet.grid(row=2, column=1, sticky=E + W)
    lt = Label(fmet, text="", foreground="black").grid(row=4, column=1, sticky=E + W) 
    lt = Label(fmet, text="Choose your method:", foreground="black",font = ("Helvetica", "12", "underline")).grid(row=5, column=1,ipady=2, sticky=E + W)   
    Radiobutton (fmet, text="All Daughter HPs", variable=HPPhaseMethod, value=0).grid (row=6, column=1, sticky=W)
    Radiobutton (fmet, text="Only Daughter 1 HPs // daughter 2 HPs ", variable=HPPhaseMethod, value=3).grid (row=6, column=4, sticky=W)

    fcol = Frame(fHP)
    fcol.grid(row=6, column=2, ipadx=1, sticky=E + W)
    lt = Label(fcol, text="", foreground="black").grid(row=1, column=1, sticky=E + W)
    lt = Label(fcol, text="Daughter HP", foreground="black").grid(row=2, column=1, sticky=E + W)
    lt = Label(fcol, text=" Color", foreground="black").grid(row=2, column=3, sticky=E + W)
    Radiobutton (fcol, text="Black", variable=HPdColorMethod, value=0).grid (row=2, column=4, sticky=W)
    Radiobutton (fcol, text="White", variable=HPdColorMethod, value=1).grid (row=2, column=5, sticky=W)
    Radiobutton (fcol, text="Operator", variable=HPdColorMethod, value=2).grid (row=2, column=6, sticky=W)
    Radiobutton (fcol, text="Blue", variable=HPdColorMethod, value=3).grid (row=2, column=7, sticky=W)
    Radiobutton (fcol, text="Red", variable=HPdColorMethod, value=4).grid (row=2, column=8, sticky=W)
    Radiobutton (fcol, text="Green", variable=HPdColorMethod, value=5).grid (row=2, column=9, sticky=W)
    lt = Label(fcol, text="     Width", foreground="black").grid(row=2, column=10, sticky=E + W)
    entry_widthHPd = Entry(fcol, width=3, foreground="blue")
    entry_widthHPd.insert(0, 1)
    entry_widthHPd.grid(row=2, column=11, sticky=E + W)

    fmode = Frame(fHP)
    fmode.grid(row=8, column=2, ipadx=2, sticky=E + W)
    lt = Label(fmode, text="Action: ", foreground="black", font = ("Helvetica", "12", "underline")).grid(row=2, column=1, sticky=E + W)
    #Radiobutton (fmode, text="None", variable=HPModeMethod, value=0).grid (row=2, column=2, sticky=W)
    Radiobutton (fmode, text=" Click to Select & Click to Place", variable=HPModeMethod, value=1).grid (row=3, column=2, sticky=W)
    Radiobutton (fmode, text="Click to Select & Click to Duplicate", variable=HPModeMethod, value=2).grid (row=4, column=2, sticky=W)
    Radiobutton (fmode, text=" Click to Select & Press +/- to scale up or r/g/b/y/p to change the color, and <Return> to complete", variable=HPModeMethod, value=3).grid (row=5, column=2, sticky=W)
    Radiobutton (fmode, text="Click to Select & Click to Delete", variable=HPModeMethod, value=4).grid (row=6, column=2, sticky=W)
    Radiobutton (fmode, text="Click to Select & Click to Write the Plane Label in the Image", variable=HPModeMethod, value=5).grid (row=7, column=2, sticky=W)


    def ShowHPs():
        global canvas, imageTk, imageCV,image, listHPs, tolHP,nbOp

        all_hp = canvas.find_withtag("HP")
        for hp in all_hp: canvas.delete(hp)

        all_hp = canvas.find_withtag("wHP")
        for hp in all_hp: canvas.delete(hp)
        
        def attrD():
            TabG = TabGrains
            Group = GroupeRotD
            TM = TMD
            TS = TSD
            h = string.atoi(entry_paramHPd[0].get())
            k = string.atoi(entry_paramHPd[1].get())
            l = string.atoi(entry_paramHPd[2].get())
            width = string.atoi(entry_widthHPd.get())
            col = "black"
            ncol = HPdColorMethod.get()
            if ncol == 0: col = "black"
            elif ncol ==1: col = "white"
            elif ncol ==2: col = "operator"
            elif ncol ==3: col = "blue"
            elif ncol ==4: col = "red"
            elif ncol ==5: col = "green"
            HP = array([h, k, l])
            return HP,TabG,Group,TM,TS,col,width
        
        def attrP(): # actually it should be called attrD2 but I kept the same name as of the habit planea
            TabG = TabGrains
            Group = GroupeRotD
            TM = TMD
            TS = TSD
            h = string.atoi(entry_paramHPp[0].get())
            k = string.atoi(entry_paramHPp[1].get())
            l = string.atoi(entry_paramHPp[2].get())
            width = string.atoi(entry_widthHPd.get())
            col = "black"
            ncol = HPdColorMethod.get()
            if ncol == 0: col = "black"
            elif ncol ==1: col = "white"
            elif ncol ==2: col = "operator"
            elif ncol ==3: col = "blue"
            elif ncol ==4: col = "red"
            elif ncol ==5: col = "green"
            HP = array([h, k, l])
            return HP,TabG,Group,TM,TS,col,width

        s = entry_nbop.get()
        nbOp = map(string.atoi,s.split(","))
        method = HPPhaseMethod.get()
        if method==0:
            infoHP = attrD()
            PlotHPs(infoHP)
        elif method==3:
            infoHPd = attrD()
            infoHPp = attrP() # actually here is NOT p for parent but it is the daughter 2
            tolHP = string.atof(entry_tolHP.get())
            PlotParallelHPs(infoHPd,infoHPp)
        

    def PlotHPs(infoHP):
        global canvas, imageTk, imageCV,image, listHPs

        HP,TabG,Group,TM,TS,col,width = infoHP
        if col=="operator": col = COLORS_OP[nbOp[0]+1]
        listHPs=[]
        listHPs.append(HP)
        for gi in Group:
            HPi = dot(transpose(gi),HP)
            flag = 1
            for HPj in listHPs:
                if HPi.tolist()== HPj.tolist() or HPi.tolist()== (-HPj).tolist(): flag = 0
            if flag : listHPs.append(HPi)
        for gr in TabG:
            w = 0.5*Scale*sqrt(len(gr.listpix))
            if GRID != "Hexagon": xt, yt = int((gr.x / Xmax) * nx * Scale), int((gr.y / Ymax) * ny * Scale)
            else: xt, yt = int((gr.x / Xmax) * nx * Scale), int((gr.y / Ymax) * ny * Scale * sqrt(3)/2)
            m_CtoS = EulersToMat(gr.euler1, gr.euler2, gr.euler3,SourceFile) #transf matrix crystal ref cubique to sample
            m_CtoS = dot(inv(TS), m_CtoS) #transf matrix crystal to sample
            normsurfC = dot(m_CtoS,array([0,0,1]))
            gsurfC = dot(TM, normsurfC)
            for HPi in listHPs:
                lineC = cross(gsurfC,HPi)
                lineS = dot(inv(m_CtoS),lineC)
                xl,yl = lineS[0],lineS[1]
                jl,il = w*normalize(array([xl,yl]))
                if GRID != "Hexagon": i1,j1,i2,j2 = xt - il, yt - jl, xt + il, yt + jl
                else: i1,j1,i2,j2 = xt - xl, yt - yl, xt + xl, yt + yl
                HPTrace = canvas.create_line(i1,j1,i2,j2, fill=col, width = width)
                canvas.itemconfig(HPTrace, tags=("HP",str(HPi)))
       

    def PlotParallelHPs(infoHPd,infoHPp):
        global canvas, imageTk, imageCV,image
        
        HPd,TabGd,Groupd,TMd,TSd,cold,widthd = infoHPd
        HPp,TabGp,Groupp,TMp,TSp,colp,widthp = infoHPp
        if cold=="operator":
            cold = COLORS_OP[nbOp[0]+1]
            colp = COLORS_OP[nbOp[0]+1]
        
        listHPd=[HPd]
        listHPp=[HPp]
        if HPd.tolist()== HPp.tolist(): sameplane=1
        else: sameplane =0
        for gi in Groupd:
            HPi = dot(transpose(gi),HPd)
            flag = 1
            for HPj in listHPd:
                if HPi.tolist()== HPj.tolist() or HPi.tolist()== (-HPj).tolist(): flag = 0
            if flag : listHPd.append(HPi)
        for gi in Groupp:
            HPi = dot(transpose(gi),HPp)
            flag = 1
            for HPj in listHPp:
                if HPi.tolist()== HPj.tolist() or HPi.tolist()== (-HPj).tolist(): flag = 0
            if flag : listHPp.append(HPi)

        for indgrd in range(1,len(TabGd)+1):
            grd = TabGd[indgrd-1]
            DtoS = EulersToMat(grd.euler1, grd.euler2, grd.euler3,SourceFile) #transf matrix crystal ref cubique to sample
            DtoS = dot(invTSD, DtoS) #transf matrix crystal to sample
            StoD = inv(DtoS)
            StoDE = inv(transpose(StoD))
            listHPdinS = [dot(StoDE,hp) for hp in listHPd]
            w = 0.5*Scale*sqrt(len(grd.listpix))
            if GRID != "Hexagon": xt, yt = int((grd.x / Xmax) * nx * Scale), int((grd.y / Ymax) * ny * Scale)
            else: xt, yt = int((grd.x / Xmax) * nx * Scale), int((grd.y / Ymax) * ny * Scale * sqrt(3)/2)
            for indgrd2 in grd.listgrvois:
                grd2 = TabGrains[indgrd2 - 1]
                nopvoisgrd = test(disorientationgrain(indgrd, indgrd2), tolHP) #liste des operateurs possibles
                if nopvoisgrd: nopvoisgrd = nopvoisgrd[0]
                else: nopvoisgrd = -1
                if nopvoisgrd in nbOp:
                    w2 = 0.5*Scale*sqrt(len(grd2.listpix))
                    if GRID != "Hexagon": xt2, yt2 = int((grd2.x / Xmax) * nx * Scale), int((grd2.y / Ymax) * ny * Scale)
                    else: xt2, yt2 = int((grd2.x / Xmax) * nx * Scale), int((grd2.y / Ymax) * ny * Scale * sqrt(3)/2)
    ##                xt = xt + int(0.3*(xt2-xt))
    ##                yt = yt + int(0.3*(yt2-yt)) #shift the trace toward the neigh grain d2
                    D2toS = EulersToMat(grd2.euler1, grd2.euler2, grd2.euler3,SourceFile) #S sample, grd daughter grains
                    D2toS = dot(invTSD, D2toS)
                    StoD2 = inv(D2toS)
                    StoD2E = inv(transpose(StoD2))
                    listHPd2inS = [dot(StoD2E,hp) for hp in listHPp]
                    
                    i=0
                    for HPd1S in listHPdinS:
                        j=0
                        for HPd2S in listHPd2inS:
                            if abs(angle(HPd1S,HPd2S))< tolHP or abs(abs(angle(HPd1S,HPd2S))-180)< tolHP: # angular tolerance = 5 deg by default
                                lineS = cross(array([0,0,1]),HPd1S)
                                xl,yl = lineS[0],lineS[1]
                                jl,il = sqrt(w*w2)*normalize(array([xl,yl]))
                                if GRID != "Hexagon": i1,j1,i2,j2 = xt - il, yt - jl, xt + il, yt + jl
                                else: i1,j1,i2,j2 = xt - xl, yt - yl, xt + xl, yt + yl
                                HPTrace = canvas.create_line(i1,j1,i2,j2, fill=cold, width = widthd)
                                canvas.itemconfig(HPTrace, tags=("HP",str(listHPd[i])))
                                if not sameplane:
                                    lineS2 = cross(array([0,0,1]),HPd2S)
                                    x1,y1 = lineS2[0],lineS2[1]
                                    jl,il = w2*normalize(array([xl,yl]))
                                    if GRID != "Hexagon": i1,j1,i2,j2 = xt2 - il, yt2 - jl, xt2 + il, yt2 + jl
                                    else: i1,j1,i2,j2 = xt2 - xl, yt2 - yl, xt2 + xl, yt2 + yl
                                    HPTrace = canvas.create_line(i1,j1,i2,j2, fill=cold, width = widthd)
                                    canvas.itemconfig(HPTrace, tags=("HP",str(listHPp[j])))
                                break
                            j+=1 
                        i+=1            
                    
    lt = Label(fHP, text="", foreground="black").grid(row=9, column=2, sticky=E + W)
    b = Button(fHP, text="Show Junction Planes", command=ShowHPs)
    b.grid(row=10, column=2, sticky=E + W)
    b["foreground"] = "blue"
    b["background"] = "grey"
    lt = Label(fHP, text="", foreground="black").grid(row=11, column=2, sticky=E + W)
    
#===================================================================================================   

def magnifymap():
    
    global OptShowA, imageTk, imageCV, image, canvas, ScaleBar, Units
    
    nx = Nbpixx
    ny = Nbpixy

    EtatRun.set("\n"+"Please wait..."+"\n")
    LabelRun["fg"] = "red"

    canvas.delete(ALL)
##    canvas["height"]= min(int(ny*Scale + 20),HMAX)
##    canvas["width"]= min(int(nx*Scale),WMAX)
    if GRID != "Hexagon": image = image.resize((int(nx * Scale), int(ny * Scale))) 
    else: image = image.resize((int(nx * Scale), int(ny * Scale * sqrt(3) / 2)))
    imageTk = ImageTk.PhotoImage(image)
    if GRID != "Hexagon": imageCV = canvas.create_image(int(nx * Scale / 2), int(ny * Scale / 2), image=imageTk)
    else: imageCV = canvas.create_image(int(nx * Scale / 2), int(ny * (sqrt(3) / 2)* Scale / 2), image=imageTk)
    canvas.itemconfig(imageCV, tags=("image"))

    dLmin = nx
    L = nx
    listL = [0.01,0.05,0.1,0.2,0.5,1,2,5,10,20,30,40,50,100,200,300,400,500,1000,2000,3000,4000,5000,10000]
    for Li in listL:
        ix = int(Li/XStep)
        dL = abs(nx/3-ix)
        if dL<dLmin:
            L = Li
            dLmin = dL
    ScaleBar = int(L / XStep)
        
    canvas["height"] = min(int(ny * Scale + 40), HMAX - 200)
    canvas["width"] = min(int(nx * Scale), WMAX - 500)
    scanH = max(int(ny * Scale), int(1.5 * ny * Scale))
    scanW = max(int(nx * Scale), int(1.5 * nx * Scale))
    canvas["scrollregion"] = (0, 0, scanW, scanH)

    if GRID != "Hexagon": canvas.create_line(int(3 * Scale), int((ny + 5) * Scale), int((ScaleBar + 3) * Scale), int((ny + 5) * Scale), width = max(int(Scale), 2), fill="Black")
    else: canvas.create_line(int(3 * Scale), int((ny + 5) * Scale *(sqrt(3) / 2)), int((ScaleBar + 3) * Scale), int((ny + 5) * Scale *(sqrt(3) / 2)), width = max(int(Scale), 2), fill="Black")
    if Scale < 1: ny = ny + int(5. / Scale)

    if GRID != "Hexagon": canvas.create_text(int((3+ScaleBar/2) * Scale), int((ny + 25) * Scale), text=str(L) + " " + Units, fill="black", font="arial " + str(int(SBSize * Scale)), anchor="center")
    else: canvas.create_text(int((3+ScaleBar/2) * Scale), int((ny + 25) * Scale *(sqrt(3) / 2)), text=str(L) + " " + Units, fill="black", font="arial " + str(int(SBSize * Scale)), anchor="center")

    if DaughterGB.get() and TabGrains: showGBD()
    if ParentGB.get() and TabParentGrains: showGBP()

    EtatRun.set("\n"+"Map ready"+"\n")
    LabelRun["fg"] = "blue"
    try: winsound.PlaySound("!", winsound.SND_ALIAS)
    except: pass 

    return

#=================================================================================================

def remspikes ():
    # attention boucle iterative dedans
    
    global TabPix
    global NextStep

    EtatRun.set("\n"+"Supressing spikes..."+"\n")
    LabelRun["fg"] = "red"

    angtol = AngTolGrain
    flag = 0
    i = 0
    stepg = Longueur/StProg
    #for i in range [lenTot]:
    while i < Longueur:
        if not i%stepg: UpDateProgress((i*100)/Longueur)
        if i > Nbpixx and i < Longueur - Nbpixx and i % Nbpixx != 0 and (i + 1) % Nbpixx != 0:
            pixi = TabPix[i]
            pixa = TabPix[i - 1]
            pixb = TabPix[i + 1]
            pixc = TabPix[i - Nbpixx]
            pixd = TabPix[i + Nbpixx]
            i1, i2, i3 = pixi[0], pixi[1], pixi[2] #les 3 angles euler
            a1, a2, a3 = pixa[0], pixa[1], pixa[2]
            b1, b2, b3 = pixb[0], pixb[1], pixb[2]
            c1, c2, c3 = pixc[0], pixc[1], pixc[2]
            d1, d2, d3 = pixd[0], pixd[1], pixd[2]
            conda = abs(i1 - a1) > angtol or abs(i2 - a2) > angtol or abs(i3 - a3) > angtol
            condb = abs(i1 - b1) > angtol or abs(i2 - b2) > angtol or abs(i3 - b3) > angtol
            condc = abs(i1 - c1) > angtol or abs(i2 - c2) > angtol or abs(i3 - c3) > angtol
            condd = abs(i1 - d1) > angtol or abs(i2 - d2) > angtol or abs(i3 - d3) > angtol

            if conda and condb and condc and condd:
                flag = 1
                r = random.randint(1, 4)
                if r == 1 : pixelnext = TabPix[i + 1]
                elif r == 2 : pixelnext = TabPix[i - 1]
                elif r == 3 : pixelnext = TabPix[i - Nbpixx]
                elif r == 4 : pixelnext = TabPix[i + Nbpixx]
                TabPix[i][0] = pixelnext[0]
                TabPix[i][1] = pixelnext[1]
                TabPix[i][2] = pixelnext[2]
        i+=1
    if flag :
        UpDateProgress(0)
        remspikes ()
    else:
        if not HideFlag.get(): showmap()

        EtatRun.set("\n"+"Next step: Remove Tips or Find the Daughter Grains"+"\n")
        LabelRun["fg"] = "blue"
        UpDateProgress(0)
        frameFile.update()
        NextStep += 1

#=================================================================================================
def remtips():
    
    global TabPix

    EtatRun.set("\n"+"Suppressing tips..."+"\n")
    LabelRun["fg"] = "red"
    frameFile.update()

    remtipshoriz()
    remtipsvert()

    if not HideFlag.get():showmap()

    EtatRun.set("\n"+"Next step: Erode or Find the daughter grains"+"\n")
    LabelRun["fg"] = "blue"
    frameFile.update()

#=================================================================================================

def remtipshoriz ():
    
    global TabPix

    angtol = 2*AngTolGrain 

    i = 0
    stepg = Longueur/StProg
    while i < Longueur - 1:
        if not i%stepg: UpDateProgress((i * 100) / Longueur)
        if i > Nbpixx + 1 and i < Longueur - Nbpixx - 1 and i % Nbpixx != 0 and (i + 1) % Nbpixx != 0:
            flag = 0
            i1, i2, i3 = TabPix[i][0], TabPix[i][1], TabPix[i][2]
            a1, a2, a3 = TabPix[i - 1][0], TabPix[i - 1][1], TabPix[i - 1][2]
            b1, b2, b3 = TabPix[i + 1][0], TabPix[i + 1][1], TabPix[i + 1][2]
            c1, c2, c3 = TabPix[i - Nbpixx][0], TabPix[i - Nbpixx][1], TabPix[i - Nbpixx][2]
            d1, d2, d3 = TabPix[i + Nbpixx][0], TabPix[i + Nbpixx][1], TabPix[i + Nbpixx][2]
            e1, e2, e3 = TabPix[i + 2][0], TabPix[i + 2][1], TabPix[i + 2][2]
            f1, f2, f3 = TabPix[i - Nbpixx + 1][0], TabPix[i - Nbpixx + 1][1], TabPix[i - Nbpixx + 1][2]
            g1, g2, g3 = TabPix[i + Nbpixx + 1][0], TabPix[i + Nbpixx + 1][1], TabPix[i + Nbpixx + 1][2]

            conda = abs(i1 - a1) > angtol or abs(i2 - a2) > angtol or abs(i3 - a3) > angtol
            condb = abs(i1 - b1) > angtol or abs(i2 - b2) > angtol or abs(i3 - b3) > angtol
            condc = abs(i1 - c1) > angtol or abs(i2 - c2) > angtol or abs(i3 - c3) > angtol
            condd = abs(i1 - d1) > angtol or abs(i2 - d2) > angtol or abs(i3 - d3) > angtol
            conde = abs(i1 - e1) > angtol or abs(i2 - e2) > angtol or abs(i3 - e3) > angtol
            condf = abs(i1 - f1) > angtol or abs(i2 - f2) > angtol or abs(i3 - f3) > angtol
            condg = abs(i1 - g1) > angtol or abs(i2 - g2) > angtol or abs(i3 - g3) > angtol

            if (conda and condc and condd and conde and condf and condg) and not condb:
                r = random.randint(1, 3)
                if r == 1 : pixelnext = TabPix[i - 1]
                elif r == 2 : pixelnext = TabPix[i + Nbpixx]
                elif r == 3 : pixelnext = TabPix[i - Nbpixx]
                TabPix[i][0] = pixelnext[0]
                TabPix[i][1] = pixelnext[1]
                TabPix[i][2] = pixelnext[2]
                TabPix[i + 1][0] = pixelnext[0]
                TabPix[i + 1][1] = pixelnext[1]
                TabPix[i + 1][2] = pixelnext[2]
        i+=1
    UpDateProgress(0)

#=================================================================================================

def remtipsvert ():
    
    global TabPix

    angtol = 2*AngTolGrain 

    i = 0
    stepg = Longueur/StProg
    while i < Longueur - 1:
        if not i%stepg: UpDateProgress((i * 100) / Longueur)
        if i > Nbpixx + 1 and i < Longueur - 2 * Nbpixx - 1 and i % Nbpixx != 0 and (i + 1) % Nbpixx != 0:
            flag = 0
            i1, i2, i3 = TabPix[i][0], TabPix[i][1], TabPix[i][2]
            a1, a2, a3 = TabPix[i - 1][0], TabPix[i - 1][1], TabPix[i - 1][2]
            b1, b2, b3 = TabPix[i + 1][0], TabPix[i + 1][1], TabPix[i + 1][2]
            c1, c2, c3 = TabPix[i - Nbpixx][0], TabPix[i - Nbpixx][1], TabPix[i - Nbpixx][2]
            d1, d2, d3 = TabPix[i + Nbpixx][0], TabPix[i + Nbpixx][1], TabPix[i + Nbpixx][2]
            e1, e2, e3 = TabPix[i + 2 * Nbpixx][0], TabPix[i + 2 * Nbpixx][1], TabPix[i + 2 * Nbpixx][2]
            f1, f2, f3 = TabPix[i + Nbpixx - 1][0], TabPix[i + Nbpixx - 1][1], TabPix[i + Nbpixx - 1][2]
            g1, g2, g3 = TabPix[i + Nbpixx + 1][0], TabPix[i + Nbpixx + 1][1], TabPix[i + Nbpixx + 1][2]

            conda = abs(i1 - a1) > angtol or abs(i2 - a2) > angtol or abs(i3 - a3) > angtol
            condb = abs(i1 - b1) > angtol or abs(i2 - b2) > angtol or abs(i3 - b3) > angtol
            condc = abs(i1 - c1) > angtol or abs(i2 - c2) > angtol or abs(i3 - c3) > angtol
            condd = abs(i1 - d1) > angtol or abs(i2 - d2) > angtol or abs(i3 - d3) > angtol
            conde = abs(i1 - e1) > angtol or abs(i2 - e2) > angtol or abs(i3 - e3) > angtol
            condf = abs(i1 - f1) > angtol or abs(i2 - f2) > angtol or abs(i3 - f3) > angtol
            condg = abs(i1 - g1) > angtol or abs(i2 - g2) > angtol or abs(i3 - g3) > angtol

            if (conda and condb and condc and conde and condf and condg) and not condd:
                r = random.randint(1, 3)
                if r == 1 : pixelnext = TabPix[i - 1]
                elif r == 2 : pixelnext = TabPix[i + 1]
                elif r == 3 : pixelnext = TabPix[i - Nbpixx]
                TabPix[i][0] = pixelnext[0]
                TabPix[i][1] = pixelnext[1]
                TabPix[i][2] = pixelnext[2]
                TabPix[i + Nbpixx][0] = pixelnext[0]
                TabPix[i + Nbpixx][1] = pixelnext[1]
                TabPix[i + Nbpixx][2] = pixelnext[2]
        i+=1

    UpDateProgress(0)

#===================================================================================================

def erode ():
    
    global TabPix

    EtatRun.set("\n"+"Erosion. Running..."+"\n")
    LabelRun["fg"] = "red"

    angtol = AngTolGrain 

    i = 0
    stepg = Longueur/StProg
    while i < Longueur:
        if not i%stepg: UpDateProgress((i * 100) / Longueur)
        if i > Nbpixx and i < Longueur - Nbpixx and i % Nbpixx != 0 and (i + 1) % Nbpixx != 0:
            pixi = TabPix[i]
            pixa = TabPix[i - 1]
            pixb = TabPix[i + 1]
            pixc = TabPix[i - Nbpixx]
            pixd = TabPix[i + Nbpixx]
            i1, i2, i3 = pixi[0], pixi[1], pixi[2] #les 3 angles euler
            a1, a2, a3 = pixa[0], pixa[1], pixa[2]
            b1, b2, b3 = pixb[0], pixb[1], pixb[2]
            c1, c2, c3 = pixc[0], pixc[1], pixc[2]
            d1, d2, d3 = pixd[0], pixd[1], pixd[2]
            conda = abs(i1 - a1) > angtol or abs(i2 - a2) > angtol or abs(i3 - a3) > angtol
            condb = abs(i1 - b1) > angtol or abs(i2 - b2) > angtol or abs(i3 - b3) > angtol
            condc = abs(i1 - c1) > angtol or abs(i2 - c2) > angtol or abs(i3 - c3) > angtol
            condd = abs(i1 - d1) > angtol or abs(i2 - d2) > angtol or abs(i3 - d3) > angtol
            flag = 0
            if condb and condc and condd and not conda:
                r = random.randint(1, 3)
                if r == 1 : pixelnext = TabPix[i + 1]
                elif r == 2 : pixelnext = TabPix[i + Nbpixx]
                elif r == 3 : pixelnext = TabPix[i - Nbpixx]
                flag = 1
            elif conda and condb and condc and not condd:
                r = random.randint(1, 3)
                if r == 1 : pixelnext = TabPix[i - 1]
                elif r == 2 : pixelnext = TabPix[i + 1]
                elif r == 3 : pixelnext = TabPix[i - Nbpixx]
                flag = 1
            elif conda and condb and condd and not condc:
                r = random.randint(1, 3)
                if r == 1 : pixelnext = TabPix[i + 1]
                elif r == 2 : pixelnext = TabPix[i + Nbpixx]
                elif r == 3 : pixelnext = TabPix[i - 1]
                flag = 1
            elif conda and condc and condd and not condb:
                r = random.randint(1, 3)
                if r == 1 : pixelnext = TabPix[i - 1]
                elif r == 2 : pixelnext = TabPix[i + 1]
                elif r == 3 : pixelnext = TabPix[i - Nbpixx]
                flag = 1
            if flag:
                TabPix[i][0] = pixelnext[0]
                TabPix[i][1] = pixelnext[1]
                TabPix[i][2] = pixelnext[2]
        i+=1

    if not HideFlag.get(): showmap()

    EtatRun.set("\n"+"Next step: Find the Daughter Grains"+"\n")
    LabelRun["fg"] = "blue"

    UpDateProgress(0)
    frameFile.update()

#===================================================================================================

def dilate0 ():
    # suppress residual or unindexed pixels
    # not iterative
    
    global TabPix,TabPixP

    if not DilatationRecurs:
        EtatRun.set("\n"+"Dilatation. Running..."+"\n")
        LabelRun["fg"] = "red"
    
    stepg = Longueur/StProg
    listrandom = range(Longueur)
    random.shuffle(listrandom)
    listPixToChange = []
    compt = 0
        
    for i in listrandom:
        compt+=1
        if not compt%stepg: UpDateProgress((compt * 100) / Longueur)
        pixi = TabPix[i]
        i1, i2, i3,bc = pixi[0], pixi[1], pixi[2], pixi[5] #les 3 angles euler et le bc qui code aussi si retained parent (val =300)
        if (i1 == 0 and bc !=300) or (i1 == 0 and  DilatationRecurs==2):
            lindpixv = Lvoisinspix4(i)
            lpixve1 = [TabPix[indpix][0] for indpix in lindpixv] # les pixels non indexes ont un euler1 = 0
            lconde1 = [e!=0 for e in lpixve1]
            nbpixind = sum(lconde1)
            if nbpixind>0:
                rr = random.randint(0, nbpixind-1)
                indrr = 0
                j=0
                for e in lconde1:
                    if e==1:
                        if j==rr: break
                        j+=1
                    indrr+=1
                pixelnext = TabPix[lindpixv[indrr]]
                listPixToChange.append([i,pixelnext])

    for ipix in listPixToChange:
        i,pix = ipix
        TabPix[i][0] = pix[0]
        TabPix[i][1] = pix[1]
        TabPix[i][2] = pix[2]
        TabPix[i][5] = pix[5]
    for ipix in listPixToChange:
        i,pix = ipix
        TabPixP[i][0] = 0
        TabPixP[i][1] = 0
        TabPixP[i][2] = 0
        TabPixP[i][5] = 310

    if DilatationRecurs==1 and len(listPixToChange)>10:
        UpDateProgress(0)
        if not HideFlag.get(): showmap()
        EtatRun.set("\n"+"Dilatation running... Remaining " + str(len(listPixToChange))+ " zeros.\n")
        LabelRun["fg"] = "red"
        frameFile.update()
    else:
        UpDateProgress(0)
        if not HideFlag.get(): showmap()
        EtatRun.set("\n"+"Next step: Find the Daughter Grains"+"\n")
        LabelRun["fg"] = "blue"
        frameFile.update()

    return len(listPixToChange)
    
#===================================================================================================

def dilate():
    global DilatationRecurs
    DilatationRecurs = 1
    # suppress unindexed pixels
    # iterative

    while dilate0()>10: dilate0()

#===================================================================================================

def dilate0_onAll():
    global TabPix, TabPixP, DilatationRecurs

    DilatationRecurs = 2         
    dilate0()
    
#===================================================================================================

def dilate_onAll():
    global TabPix, TabPixP

    TabPixP = []
    for i in range(Longueur):
        TabPixP.append([0, 0, 0, 0, 0, 310])
    for i in range(Longueur):
        if TabPix[i][0]== 0: TabPix[i][5]=310          
    dilate()
    
#===================================================================================================
        
def CalcMapGOS(): #affiche que les angles
    global gosmin, gosmax, OptShowA, OptShowB, GOSmean, TabGrains

    if 0:
        m = "info\n"
        tkMessageBox.showinfo("Warning", m)
        return
    else:
        gosmin, gosmax = 180,0
        EtatRun.set("\n"+"Calculating Angular Deformation Map (GOS = Grain Orientation Spreading)..."+"\n")
        LabelRun["fg"] = "red"
        frameFile.update()
        LL = len(TabGrains)
        stepg = max(LL/StProg,1)
        compt = 0
        for grd in TabGrains:
            Egrd = grd.euler1,grd.euler2,grd.euler3
            invMatEgrd = inv(EulersToMat(Egrd[0],Egrd[1],Egrd[2],SourceFile))
            ang = 0
            n=0
            compt+=1
            if not compt%stepg: UpDateProgress((compt * 100) / LL)
            for i in grd.listpix:
                Ei = array([TabPix[i][0], TabPix[i][1], TabPix[i][2]])
                m = dot(EulersToMat(Ei[0],Ei[1],Ei[2],SourceFile), invMatEgrd)
                mc = dot(inv(TSD), dot(m, TSD))
                a = testIdang(mc)
                #a, axe  = calculeRot(mc) donnerait aussi l axe de rotation
                if a and a <= AngTolGrain:
                    ang+=a
                    n+=1
            if n>0:
                angmean= ang/n
                grd.gos = round(angmean,7)
                if angmean>gosmax: gosmax = angmean
                if angmean<gosmin: gosmin = angmean
            
        GOSmean = 0.0000000001
        for grd in TabGrains:
            GOSmean+=grd.gos
            
        GOSmean = round(GOSmean/len(TabGrains),7)
        print "GOS mean = ", GOSmean
        
        UpDateProgress(0)
        OptShowB = "graind"
        OptShowA = "gos"
            
        ShowMapGOS(1)
        
#===================================================================================================
        
def CalcMapDef(): #affiche que les angles, il faudra afficher les axes
    global TabPixDef, defmin, defmax, OptShowA,OptShowB, KAMmean, TabGrains, AngTolGrain 

    if AngTolGrain <10:
        n= "Please note that the KAM will be calculated inside each grain and thus will be limited to the angle chosen for the grain boundaries. \n"
        o= "If required, the tolerance angle could be increased to higher values by changing this angle (Tolerance for grain identification) and then using Ignore Low Misorientation" 
        tkMessageBox.showinfo("Warning", n+o)
##        AngTolGrain = 10
##        EtatAngTolGrain.set(AngTolGrain)
##        IgnoreMisorientedGrainsD()
        
    if 1:
        TabPixDef = [0 for i in range (len(TabPix))]

        EtatRun.set("\n"+"Calculating Angular Deformation Map (KAM = Kernel Average Misorientation)..."+"\n")
        LabelRun["fg"] = "red"
        frameFile.update()

        LL = len(TabGrains)
        stepg = max(LL/StProg,1)
        compt = 0
        for grd in TabGrains:
            angmean = 0
            n2 =0
            compt+=1
            if not compt%stepg: UpDateProgress((compt * 100) / LL)
            for i in grd.listpix:
                ang = 0
                Ei = array([TabPix[i][0], TabPix[i][1], TabPix[i][2]])
                invMatEi = inv(EulersToMat(Ei[0],Ei[1],Ei[2],SourceFile)) 
                n=0
                for iv in Lvoisinspix4(i):
                    if TabPix[iv][3]==compt:
                        Eulers = array([TabPix[iv][0], TabPix[iv][1], TabPix[iv][2]])
                        m = dot(EulersToMat(Eulers[0],Eulers[1],Eulers[2],SourceFile), invMatEi)
                        mc = dot(inv(TSD), dot(m, TSD))
                        a = testIdang(mc)
                        #a, axe  = calculeRot(mc) donnerait aussi l axe de rotation
                        if a and a< AngTolGrain:
                            ang+=a
                            n+=1
                if n:
                    ang = ang/n
                    angmean+= ang
                    n2+=1
                else: ang = 0
                TabPixDef[i] = ang
            if n2 and angmean:
                angmean = angmean / n2
                grd.kam = round(angmean,7)
            
        defmin, defmax = 180,0
        for ang in TabPixDef:
            if ang>defmax: defmax = ang
            if ang<defmin: defmin = ang
            
        KAMmean = 0
        for grd in TabGrains:
            KAMmean+=grd.kam
            
        KAMmean = round(KAMmean/len(TabGrains),7)
        print "KAM mean = ", KAMmean
        
        UpDateProgress(0)
        OptShowB == "graind"
        OptShowA = "def"
            
        ShowMapDef(1)
 #===================================================================================================
        
def CalcMapDef2(): #affiche les angles et les axes mais plus long
    global TabPixDef, defmin, defmax, OptShowA, OptShowB, KAMmean, TabGrains, AngTolGrain

    if AngTolGrain <10:
        m = "The angular tolerance that have been chosen to identify the grains seems small for KAM \n"
        n= "Please note that the KAM will be calculated inside each grains and thus will be limited to that tolerance angle. \n"
        o= "If required, the tolerance angle could be increased to higher values by changing this angle and then using Ignore Low Misorientation" 
        tkMessageBox.showinfo("Warning", m+n+o)
##        AngTolGrain = 10
##        EtatAngTolGrain.set(AngTolGrain)
##        IgnoreMisorientedGrainsD()

    if 1:
        TabPixDef = [0 for i in range (len(TabPix))]

        EtatRun.set("\n"+"Calculating Angular Deformation Map KAM + Rotation axes..."+"\n")
        LabelRun["fg"] = "red"
        frameFile.update()
        listaxesCryst = []
        listaxesSamp = []
        LL = len(TabGrains)
        stepg = max(LL/StProg,1)
        
        compt = 0
        for grd in TabGrains:
            compt+=1
            if not compt%stepg: UpDateProgress((compt * 100) / LL)
            angmean = 0
            n2=0
            for i in grd.listpix:
                ang = 0
                invMatEi = inv(EulersToMat(TabPix[i][0], TabPix[i][1], TabPix[i][2],SourceFile)) 
                s=0
                axeCryst = []
                for iv in Lvoisinspix4(i):
                    if TabPix[iv][3]==compt:
                        m = dot(EulersToMat(TabPix[iv][0], TabPix[iv][1], TabPix[iv][2],SourceFile),invMatEi)
                        tr = trace(m)
                        if tr >3: tr = 3
                        elif tr <-1: tr=-1
                        a = abs(arccos((tr - 1) / 2)) * 180. / pi
                        if abs(a) > 2:
                            desMIN = disorientationpixelANGAXE(i,iv)
                            a = desMIN[0]
                            axeCryst = desMIN[1]
                            if a and a<AngTolGrain and len(axeCryst)>0:
                                s+=1
                                ang+=a
                                if  a>2:
                                    axeCryst= normalizeCryst(axeCryst,TMD)
                                    #print IntegriseAxe(axeCryst), a
                                    listaxesCryst.append(axeCryst)
                                    axeSamp = normalize(dot(invMatEi,axeCryst))
                                    listaxesSamp.append(axeSamp)
                        elif a<AngTolGrain:
                            s+=1
                            ang+=a
                            
                if s :
                    ang = ang/s
                    angmean+= ang
                    n2+=1
                else: ang = 0
                if ang > AngTolGrain: ang = AngTolGrain # pas tres propre
                TabPixDef[i] = ang
                
            if n2 and angmean:
                angmean = angmean/n2
                grd.kam = round(angmean,7)
                        
        defmin, defmax = 180,0
        for ang in TabPixDef:
            if ang>defmax: defmax = ang
            if ang<defmin: defmin = ang

        KAMmean = 0
        for grd in TabGrains:
            KAMmean+=grd.kam
            
        KAMmean = round(KAMmean/len(TabGrains),7)
        print "KAM mean = ", KAMmean
            
        UpDateProgress(0)
        OptShowB == "graind"
        OptShowA = "def"

        ShowMapDef(1)
            
        if len(listaxesCryst)>0:
            print "len(listaxesCryst)", len(listaxesCryst)
            t = FileName + ". KAM axes in Crystal Frame"
            EtatRun.set("\n"+"Preparing figure..."+"\n")
            LabelRun["fg"] = "red"
            frameFile.update()
            PolFigDirections (root, 600, listaxesCryst, TMD, TSD, GroupeRotD, t, "isoCryst", CrystD.syst,6) #6 maillage de la PF
            EtatRun.set("\n"+"Pole figure completed..."+"\n")
            LabelRun["fg"] = "blue"
        else: print "all the misorientation angles are below 2 degree, the rotation axes could not be identified"
        
        if len(listaxesSamp)>0:
            print "len(listaxesSamp)", len(listaxesSamp)
            t = FileName + ". KAM axes in Sample Frame"
            EtatRun.set("\n"+"Preparing figure..."+"\n")
            LabelRun["fg"] = "red"
            frameFile.update()
            PolFigDirections (root, 600, listaxesSamp, [Identity], [Identity], [Identity], t, "isoSamp",CrystD.syst,6)
            EtatRun.set("\n"+"Pole figure completed..."+"\n")
            LabelRun["fg"] = "blue"
        else: print "all the misorientation angles are below 2 degree, the rotation axes could not be identified"
    
#===================================================================================================
    
def ShowMapGOS(event):
    global OptShowA, OptShowB

    refreshButtons2()
    ButtonM["foreground"] = "white"
    ButtonM["background"] = "blue"
      
    if not TabGrains: IdentifyDaughterGrains()  
##        info = "The daughter grains should be identified before calculating the angular deformations"
##        tkMessageBox.showinfo("Info", info)

    if not GOSmean: CalcMapGOS()
      
    if TabGrains:
        refreshButtons1()
        OptShowA = "gos"
        OptShowB = "graind"
        ButtonGOS["foreground"] = "white"
        ButtonGOS["background"] = "blue"
        showmap()
        
#===================================================================================================
    
def ShowMapDef(event):
    global OptShowA, OptShowB

    refreshButtons2()
    ButtonM["foreground"] = "white"
    ButtonM["background"] = "blue"
    if OptShowA != "def" and OptShowA != "def2" : OptShowA = "def2"
        
    if not TabGrains: IdentifyDaughterGrains()  
##        info = "The daughter grains should be identified before calculating the angular deformations"
##        tkMessageBox.showinfo("Info", info)
        
    if not TabPixDef: CalcMapDef()
      
    if TabGrains and TabPixDef:
        refreshButtons1()
        ButtonDef["foreground"] = "white"
        ButtonDef["background"] = "blue"
        OptShowB = "graind"
        if OptShowA == "def":
            showmap()
            OptShowA = "def2" # map KAM mean value for each daughter grain
        else:
            showmap()
            OptShowA = "def" # map local KAM values       

#===================================================================================================    
    
def ShowMapTri(event):
    global OptShowA, ComptTriangle

    refreshButtons2()
    ButtonM["foreground"] = "white"
    ButtonM["background"] = "blue"

    if not TabGrains or not TabParentGrains:
        info = "The daughter grains should be identified and the parent grains should be reconstructed"
        tkMessageBox.showinfo("Info", info)
        
    elif not TabPixTriangle: CalcTriangleOR()
      
    elif TabPixTriangle:
        wd = ""
        ComptTriangle = (ComptTriangle+1)%2
        refreshButtons1()
##        ButtonTri["foreground"] = "white"
##        ButtonTri["background"] = "blue"
        if ComptTriangle:
            OptShowA = "Triangle"
            wd = "(P,KS,NW) in (R,G,B) on 10 deg scale"
        else:
            OptShowA = "Triangle2"
            wd = "P=Red, KS=Green, NW=Blue, and C =Orange within 2.5 deg. tolerance"
        EtatRun.set(ResTri+'\n'+wd+"\n")
        LabelRun["fg"] = "blue"
        frameFile.update()
        frameMenuCV.update()
        showmap()
        
        
#===================================================================================================

def ClassGrains(opt="graind"):
    
    # reordonne la table des grains par ordre inv de taille
    # normalement ca a ete fait avant, mais le pg est dur a relire
    global TabPix, TabGrains #, TabParentGrains pas encore fait pour grains parents

    if opt == "graind":
        TabG = TabGrains
        indpix = 3

    elif opt == "grainp":
        TabG = TabParentGrains # A FINIR!!
        indpix = 4
        print "ClassGrains not done for parent grains"
        return

    NewTabGrains = []
    TabClass = []

    for i in range(len(TabG)):
        gr = TabG[i]
        size = len(gr.listpix)
        TabClass.append([size, i])

    TabClass.sort(reverse=True) # classement par taille du + grand au + petit
    TabClassInd = [x[1] for x in TabClass]
    TabClassIndRev = [0 for x in TabClass]
    i = 0
    for indj in TabClassInd:
        TabClassIndRev[indj] = i
        i += 1

    j = 1
    for i in TabClassInd:
        gr = TabG[i]
        listgrvois = gr.listgrvois
        newlistgrvois = []
        for indgrvois in listgrvois:
            try:
                newindgrvois = TabClassIndRev[indgrvois - 1]
                newlistgrvois.append(newindgrvois + 1)
            except: pass
        lvbis = list(set(newlistgrvois))
        if i in lvbis: lvbis.remove(i)
        gr.listgrvois = lvbis
        gr.graind = j 
        NewTabGrains.append(gr)
        for pix in gr.listpix: TabPix[pix][indpix] = j  #renumerote les pixels juillet 2010 j+1 => j
        j+=1

    if opt == "graind": TabGrains = NewTabGrains

    return

#===================================================================================================

def CalculateGrainBoundaries(opt="graind"):
    
    # calcule les pixels qui sont au bord des grains
    # normalement ca a ete fait avant, mais le pg est dur a relire
    global TabGrains, TabParentGrains

    if opt == "graind":
        for gr in TabGrains:
            listpixfr = []
            for pix in gr.listpix:
                lv = Lvoisinspix4(pix)
                flag = 0
                for pixv in lv:
                    if TabPix[pixv][3] != gr.graind:#3=graind
                        flag = 1 
                        break
                if flag: listpixfr.append(pix)
            gr.listpixfr = listpixfr
        writeGBD()

    elif opt == "grainp":
        for gr in TabParentGrains:
            listpixfr = []
            for pix in gr.listpix:
                lv = Lvoisinspix4(pix)
                flag = 0
                for pixv in lv:
                    if TabPix[pixv][4] != gr.grainp: #4=grainp DEVRAIT MARCHER MAIS PB...
                    #if TabPix[pixv][4] != TabPix[pix][4]: # marche mais peu satisfaisant
                        flag = 1
                        break
                if flag: listpixfr.append(pix)
            gr.listpixfr = listpixfr
        writeGBP()

    elif opt == "grainr":
        i=1
        for gr in TabRetainedGrains:
            listpixfr = []
            for pix in gr.listpix:
                lv = Lvoisinspix4(pix)
                flag = 0
                for pixv in lv:
                    if TabPix[pixv][4] != -i: # negatif for retained parent
                        flag = 1
                        break
                if flag: listpixfr.append(pix)
            gr.listpixfr = listpixfr
            i+=1
        writeGBP()

#==========================================================================================================

def fctkicalgrain(liste, lpixvois):
    
    # liste = liste des pixels qui forment un grain
    # lpixvois[i] = liste des pixels formant les grains voisins de liste[i]
    # A partir de cette liste on peut refaire le Tableau Tabrecsec en numerotant les grains lineairement
    global TabGrains, TabPix

    print "Calculating grains..."
    EtatRun.set("\n"+"Nb of Grains = " + str(len(liste)) + " Please wait..."+"\n")
    LabelRun["fg"] = "red"
    TabGrains = []
    l = []
    l1 = []
    lvoisins = []

    UpDateProgress(0)
    LL = max(len(liste),1)
    stepg = max(len(liste)/StProg,1)

    for i in range(len(liste)):
        if not i%stepg: UpDateProgress((i * 100) / LL)
        ang = 0
        x, y = 0, 0
        em1, em2, em3 = 0, 0, 0
        axe1 = array([0, 0, 0])
        for j in range(len(liste[i])):
            TabPix[liste[i][j]][3] = i + 1 #3=graind
            xp, yp = PosXY(liste[i][j])
            x = x + xp
            y = y + yp
            e1, e2, e3 = TabPix[liste[i][j]][0], TabPix[liste[i][j]][1], TabPix[liste[i][j]][2]
            em1, em2, em3 = e1 + em1, e2 + em2, e3 + em3 # ose moyenner les angles d Euler
            
        lgrvois = []
        for k in lpixvois[i]:
            indgrk = Tabrecsec2[k]
            if  indgrk not in lgrvois and indgrk != i + 1:
                lgrvois.append(indgrk)

        l = len(liste[i])
        gr = Grain(1, x / l, y / l, em1 / l, em2 / l, em3 / l, i + 1, [], 0, liste[i], [], lgrvois, 30)

        TabGrains.append(gr)

    UpDateProgress(0)
#==========================================================================================================

def fctkicalgrainP(liste, lpixvois):
    
    # liste = liste des pixels qui forment un grain
    # lpixvois[i] = liste des pixels formant les grains voisins de liste[i]
    # A partir de cette liste on peut refaire le Tableau Tabrecsec en numerotant les grains lineairement
    global TabRetainedGrains, TabPixP, TabPix

    print "Calculating retained parent grains..."
    EtatRun.set("\n"+"Nb of Grains = " + str(len(liste)) + " Please wait..."+"\n")
    LabelRun["fg"] = "red"
    l = []
    l1 = []
    lvoisins = []

    UpDateProgress(0)
    LL = max(len(liste),1)
    stepg = max(len(liste)/StProg,1)

    for i in range(len(liste)):
        if not i%stepg: UpDateProgress((i * 100) / LL)
        ang = 0
        x, y = 0, 0
        em1, em2, em3 = 0, 0, 0
        axe1 = array([0, 0, 0])
        for j in range(len(liste[i])):
            TabPixP[liste[i][j]][4] = -(i + 1) #4=grainp et qd negatif veut dire retained
            TabPix[liste[i][j]][4] = -(i + 1) #4=grainp et qd negatif veut dire retained
            xp, yp = PosXY(liste[i][j])
            x = x + xp
            y = y + yp
            e1, e2, e3 = TabPixP[liste[i][j]][0], TabPixP[liste[i][j]][1], TabPixP[liste[i][j]][2]
            em1, em2, em3 = e1 + em1, e2 + em2, e3 + em3 # ose moyenner les angles d Euler
            
        lgrvois = []
        for k in lpixvois[i]:
            indgrk = Tabrecsec2P[k]
            if  indgrk not in lgrvois and indgrk != i + 1:
                lgrvois.append(indgrk)

        l = len(liste[i])
        gr = Grain(1, x / l, y / l, em1 / l, em2 / l, em3 / l, i + 1, [], 0, liste[i], [], lgrvois, 30)

        TabRetainedGrains.append(gr)

    UpDateProgress(0)

#===================================================================================================
   
def TransF(ft, ea, eb):
    r = (ea + ft[0] * eb) - ft[1]
    return r
    
#===================================================================================================
    
def FunctEulersD(teA, teB):
    e1A, e2A, e3A = teA
    e1B, e2B, e3B = teB
    res = 0
    #FEulersD[0] #liste des fonctions donnant les equivalences des angles d Eulers
    #FEulersD[1] #liste des matrices de symetries nonreductibles a des functions lineaires (pour les cubiques)
    for fEulers in FEulersD[0]:
        f1, f2, f3 = fEulers
        r1, r2, r3 = TransF(f1, e1A, e1B), TransF(f2, e2A, e2B), TransF(f3, e3A, e3B)
        if (abs(r1) + abs(r2) + abs(r3)) < AngTolGrain:
            e1Bn, e2Bn, e3Bn = (e1A - r1), (e2A - r2), (e3A - r3)
            res = e1Bn, e2Bn, e3Bn
            break
    return res
#===================================================================================================
    
def FunctEulersP(teA, teB):
    e1A, e2A, e3A = teA
    e1B, e2B, e3B = teB
    res = 0
    for fEulers in FEulersP[0]:
        f1, f2, f3 = fEulers
        r1, r2, r3 = TransF(f1, e1A, e1B), TransF(f2, e2A, e2B), TransF(f3, e3A, e3B)
        if (abs(r1) + abs(r2) + abs(r3)) < AngTolGrain:
            e1Bn, e2Bn, e3Bn = (e1A - r1), (e2A - r2), (e3A - r3)
            res = e1Bn, e2Bn, e3Bn
            break
    return res

#===================================================================================================

def IdentifyRetainedParentGrains():
    
    global OptShowB, OptShowA
    global TabPixP, TabRetainedGrains, TabGrainsRecP, iparent 
    global TabGrP, TabGrP_vois, Tabrecsec2P, listPixPtGrainsP, listgrainsP
    global listi_intraP, listfr_toexploreP, listfrtot_extP, igrain
    global compt #, nbpix

    TabGrainsRecP = []
    Tabrecsec2P = []
    TabRetainedGrains = []
    iparent = 0

    EtatRun.set("\n"+"Identity Retained Parent Grains with Euler angles. Running..."+"\n")
    LabelRun["fg"] = "red"
    frameFile.update()

    nx = Nbpixx
    ny = Nbpixy

    stepref = 0

    TabGrP = [] 
    TabGrP_vois = []
    listPixPtGrainsP = [] # liste des pixels des petits grains, a dilater a la fin
    lvoispixP = []
    GroupeRotPOrtho = []
    for g in GroupeRotP:
        GroupeRotPOrtho.append(dot(dot(TSP, g),invTSP))   
  
    #*********************************** 
    def DisorientOrtho(m):
        gm=[]
        for symk in GroupeRotPOrtho: 
            tr = trace(dot(symk,m))
            if tr < -1: tr = -1
            elif tr > 3: tr = 3
            arand = abs(arccos((tr - 1) / 2)) * 180. / pi
            if arand < AngTolGrain:
                gm = symk
                break
        return gm
            
    #*********************************** recursif
    def buildgrainP(): #listi = total des pixels, listifr = juste ceux de la frontiere a traiter
        global TabPixP, TabGrP, TabGrP_vois, Tabrecsec2P, listi_intraP, listfr_toexploreP, listfrtot_extP, igrain
        global compt, listPixPtGrainsP
        
        newlistfrP = []
        for i in listfr_toexploreP:
            Ei = array([TabPixP[i][0], TabPixP[i][1], TabPixP[i][2]])
            if SourceFile == 'TSL': invMatEi = inv(EulersToMat(Ei[0],Ei[1],Ei[2],SourceFile)) #LALA
            for iv in Lvoisinspix4(i):
                if not Tabrecsec2P[iv] and TabPixP[iv][0]!=0: 
                    Eulers = array([TabPixP[iv][0], TabPixP[iv][1], TabPixP[iv][2]])
                    d = add.reduce(map(abs, Eulers - Ei))
                    f = 0
                    if d < AngTolGrain:
                        f = 1
                    elif SourceFile == 'HKL': 
                        r = FunctEulersP(Ei, Eulers)
                        if r:
                            TabPixP[iv][0], TabPixP[iv][1], TabPixP[iv][2] = r
                            f = 1
                            # pour les cubiques cela ne suffit pas car il y a des matrices de symetries residuelles non reductible
                            # a des lois lineaires sur les Eulers
                            # nous ferons donc un IgnoreMisorientedGrains apres
                    elif SourceFile == 'TSL' : #LALA
                        r = FunctEulersP(Ei, Eulers) #r = FEulers(Ei0, Eulers) LALA2
                        if r:
                            TabPixP[iv][0], TabPixP[iv][1], TabPixP[iv][2] = r
                            f = 1
                    if f:
                        newlistfrP.append(iv)
                        Tabrecsec2P[iv] = igrain
                    else:
                        listfrtot_extP.append(iv)
                else:
                    listfrtot_extP.append(iv)

        if not newlistfrP or compt >= RecursionLim:
            #cas de petits grains
            if len(listi_intraP)<= MinRetainedSize: # min grain size for retained parent = same as for daughter grains (in pixels), but can be decorrelated from it
                for iintra in listi_intraP:
                    Tabrecsec2P[iintra] = -1
                    TabPixP[iintra][0], TabPixP[iintra][1], TabPixP[iintra][2] = 0,0,0
                igrain = igrain-1
                listPixPtGrainsP.extend(listi_intraP)
            # cas de la limite de recursion        
            else:      
                TabGrP.append(listi_intraP)
                TabGrP_vois.append(listfrtot_extP)
        else:
            compt+=1
            listi_intraP.extend(newlistfrP)
            listfr_toexploreP = newlistfrP
            buildgrainP()

        #***********************************
            
    Tabrecsec2P = [0 for i in range(Longueur)]
    for i in range(Longueur):
        if TabPixP[i][0]==0: Tabrecsec2P[i]=-1

    igrain = 0 # indice du grain
    #stepg = Longueur/StProg
    print "Start Recursion"
    while 1:
        igrain += 1
        lvoispixP = [] #liste des pixels voisins du grains qui sera identifie
        try: i = Tabrecsec2P.index(0) #indice du pixel
        except: break
        if i == len(TabPixP) - 1:break
        if not i%100: UpDateProgress((i * 100) / Longueur)
        Tabrecsec2P[i] = igrain
        listi_intraP = [i]
        listfr_toexploreP = [i]
        listfrtot_extP = []
        compt = 1
        buildgrainP() #lance la recursivite
    print "End of the Recursion"
    
    # efface les petits grains internes par dilatation
    listgrainsP = []
    def fermetureP(listpixpt):
        global TabGrP, TabPixP, Tabrecsec2P, listgrainsP
        nlist=[]
        for ipix in listpixpt:
            f=0
            lvois = Lvoisinspix4(ipix)
            random.shuffle(lvois)
            for iv in lvois:
                indg= Tabrecsec2P[iv]
                if indg>0:
                    f=1
                    Ei = array([TabPixP[iv][0], TabPixP[iv][1], TabPixP[iv][2]])
                    TabPixP[ipix][0], TabPixP[ipix][1], TabPixP[ipix][2] = Ei
                    TabGrP[indg-1].append(ipix)
                    Tabrecsec2P[ipix]=indg
                    if indg not in listgrainsP: listgrainsP.append(indg)
                    break
            if not f: nlist.append(ipix)
        if len(nlist)==len(listpixpt): return
        if nlist: fermetureP(nlist)
    random.shuffle(listPixPtGrainsP)
    fermetureP(listPixPtGrainsP)
    listPixPtGrainsP = []
    
    # recalcule frontiere des grains ayant absorbe les pts grains
    for indg in listgrainsP:
        listpixfrP = []
        for pix in TabGrP[indg-1]:
            lv = Lvoisinspix4(pix)
            for pixv in lv:
                if Tabrecsec2P[pixv]!= indg: listpixfrP.append(pixv)               
        TabGrP_vois[indg-1] = listpixfrP
        
    print "End of the dilatation of the small grains"      

    #**********************************************************************************************

    EtatRun.set("\n"+"Just with Eulers. Nb of Grains = " + str(len(TabGrD)) + "Please wait..."+"\n")
    LabelRun["fg"] = "red"

    fctkicalgrainP(TabGrP, TabGrP_vois)

    OptShowB = "euler1"

    #ClassGrains()

    #if CrystD.syst == "Cubic" and SourceFile != 'TSL': IgnoreMisorientedGrainsDRemain() #LALA
    #if CrystP.syst == "Cubic" : IgnoreMisorientedGrainsPRemain()
    
    #IgnoreSmallGrainsP() # ici ici a changer pour retainedgrain

    CalculateGrainBoundaries("grainr") # calcule les pixels qui sont en frontiere de grain

    Tabrecsec2P = []

    OptShowB = "graind"
    OptShowA = "euler1"
    refreshButtons1()
    refreshButtons2()
    ButtonM["foreground"] = "white"
    ButtonM["background"] = "blue"
    ButtonA["foreground"] = "white"
    ButtonA["background"] = "blue"

    if not HideFlag.get(): showmap()
    print "Number of retained grains", len(TabRetainedGrains)
    EtatRun.set("\n"+"Nb of Retained Grains = " + str(len(TabRetainedGrains)) + ". Next step: Reconstruct the Parents Grains"+"\n")
    LabelRun["fg"] = "blue"
    UpDateProgress(0)
    frameFile.update()
    NextStep = 5
    try: winsound.PlaySound("!", winsound.SND_ALIAS)
    except: pass
    print "End of Identification of the Retained Parent Grains"

#==========================================================================================================

def IdentifyDaughterGrains():
    
    global OptShowB, OptShowA
    global TabPix, TabGrains, TabParentGrains, TabGrainsRec, TabOrient, listNuclei, iparent
    global TabGrD, TabGrD_vois, Tabrecsec2, listPixPtGrains, listgrains
    global listi_intra, listfr_toexplore, listfrtot_ext, igrain
    global compt #, nbpix
    global NextStep, canvas

    # A partir de cette liste on peut refaire le Tableau Tabrecsec en numerotant les grains lineairement
    #Tabrecsec2=ones((Longueur,1), Float).tolist()
    #Reconstruction of the daughter grains
    #tol = angular tolerance

    # On fait une deuxieme fonction pour diminuer le max de Tabrecsec
    #Pour simplifier on s interesse au cas sans zeros

    remspikes ()

    TabParentGrains = []
    TabGrainsRec = []
    TabOrient = []
    listNuclei = []
    iparent = 0

    TimeStart = time.clock()

    if Transition == "?":
        EtatRun.set("\n"+"CHOOSE THE TRANSITION BEFORE!!"+"\n")
        LabelRun["fg"] = "red"
        frameFile.update()
        return
    else: pass

    EtatRun.set("\n"+"Daughter Grain Reconstruction with Euler angles. Running..."+"\n")
    LabelRun["fg"] = "red"
    frameFile.update()

    nx = Nbpixx
    ny = Nbpixy

    stepref = 0

    TabGrD = [] 
    TabGrD_vois = []
    listPixPtGrains = [] # liste des pixels des petits grains, a dilater a la fin
    lvoispix = []
    GroupeRotDOrtho = []
    for g in GroupeRotD:
        GroupeRotDOrtho.append(dot(dot(TSD, g),invTSD))   
  
    #*********************************** 
    def DisorientOrtho(m):
        gm=[]
        for symk in GroupeRotDOrtho: 
            tr = trace(dot(symk,m))
            if tr < -1: tr = -1
            elif tr > 3: tr = 3
            arand = abs(arccos((tr - 1) / 2)) * 180. / pi
            if arand < AngTolGrain:
                gm = symk
                break
        return gm
            
    #*********************************** recursif
    def buildgrain(): #listi = total des pixels, listifr = juste ceux de la frontiere a traiter
        global TabPix, TabGrD, TabGrD_vois, Tabrecsec2, listi_intra, listfr_toexplore, listfrtot_ext, igrain
        global compt, listPixPtGrains
        
        newlistfr = []
        for i in listfr_toexplore:
            Ei = array([TabPix[i][0], TabPix[i][1], TabPix[i][2]])
            if SourceFile == 'TSL': invMatEi = inv(EulersToMat(Ei[0],Ei[1],Ei[2],SourceFile)) #LALA
            for iv in Lvoisinspix4(i):
                #  pb qui arrivent parfois avec .ang , voir Hattal 22 oct 2019
                
                if not Tabrecsec2[iv] and TabPixP[iv]!=300: 
                    Eulers = array([TabPix[iv][0], TabPix[iv][1], TabPix[iv][2]])
                    d = add.reduce(map(abs, Eulers - Ei))
                    f = 0
                    if d < AngTolGrain:
                        f = 1
                    elif SourceFile == 'HKL': 
                        r = FunctEulersD(Ei, Eulers)
                        if r:
                            TabPix[iv][0], TabPix[iv][1], TabPix[iv][2] = r
                            f = 1
                            # pour les cubiques cela ne suffit pas car il y a des matrices de symetries residuelles non reductible
                            # a des lois lineaires sur les Eulers
                            # nous ferons donc un IgnoreMisorientedGrains apres
                    elif SourceFile == 'TSL' : #LALA
                        r = FunctEulersD(Ei, Eulers) #r = FEulers(Ei0, Eulers) LALA2
                        if r:
                            TabPix[iv][0], TabPix[iv][1], TabPix[iv][2] = r
                            f = 1
                    if f:
                        newlistfr.append(iv)
                        Tabrecsec2[iv] = igrain
                    else:
                        listfrtot_ext.append(iv)
                else:
                    listfrtot_ext.append(iv)
                

        if not newlistfr or compt >= RecursionLim:
            #cas de petits grains
            if len(listi_intra)<= MinGrainSize:
                for iintra in listi_intra:
                    Tabrecsec2[iintra] = -1
                    TabPix[iintra][0], TabPix[iintra][1], TabPix[iintra][2] = 0,0,0
                igrain = igrain-1
                listPixPtGrains.extend(listi_intra)
            # cas de la limite de recursion        
            else:      
                TabGrD.append(listi_intra)
                TabGrD_vois.append(listfrtot_ext)
        else:
            compt+=1
            listi_intra.extend(newlistfr)
            listfr_toexplore = newlistfr
            buildgrain()

        #***********************************

    if Transition == "?":
        EtatRun.set("\n"+"CHOOSE THE TRANSITION BEFORE!!"+"\n")
        LabelRun["fg"] = "red"
        frameFile.update()
        return
    else: pass

    EtatRun.set("\n"+"Decorrelate the Equivalent Euler angles. Running..."+"\n")
    LabelRun["fg"] = "red"
    frameFile.update()
            
    Tabrecsec2 = [0 for i in range(Longueur)]
    for i in range(Longueur):
        if TabPix[i][0]==0: Tabrecsec2[i]=-1

    igrain = 0 # indice du grain
    #stepg = Longueur/StProg
    print "Start Recursion"
    while 1:
        igrain += 1
        lvoispix = [] #liste des pixels voisins du grains qui sera identifie
        try: i = Tabrecsec2.index(0) #indice du pixel
        except: break
        if i == len(TabPix) - 1:break
        if not i%100: UpDateProgress((i * 100) / Longueur)
##        if SourceFile == 'TSL':#LALA2 = identifie les grains en prenant un pixel en ref absolu (et non en relatif)
##            Ei0 = array([TabPix[i][0], TabPix[i][1], TabPix[i][2]]) 
##            invMatEi0 = inv(EulersToMat(Ei0[0],Ei0[1],Ei0[2],SourceFile))
        Tabrecsec2[i] = igrain
        listi_intra = [i]
        listfr_toexplore = [i]
        listfrtot_ext = []
        compt = 1
        buildgrain() #lance la recursivite
    print "End Recursion"
    
    # efface les petits grains internes par dilatation
    listgrains = []
    def fermeture(listpixpt):
        global TabGrD, TabPix, Tabrecsec2, listgrains
        nlist=[]
        for ipix in listpixpt:
            f=0
            lvois = Lvoisinspix4(ipix)
            random.shuffle(lvois)
            for iv in lvois:
                indg= Tabrecsec2[iv]
                if indg>0:
                    f=1
                    Ei = array([TabPix[iv][0], TabPix[iv][1], TabPix[iv][2]])
                    TabPix[ipix][0], TabPix[ipix][1], TabPix[ipix][2] = Ei
                    TabGrD[indg-1].append(ipix)
                    Tabrecsec2[ipix]=indg
                    if indg not in listgrains: listgrains.append(indg)
                    break
            if not f: nlist.append(ipix)
        if len(nlist)==len(listpixpt): return
        if nlist: fermeture(nlist)
    random.shuffle(listPixPtGrains)
    fermeture(listPixPtGrains)
    listPixPtGrains = []
    
    # recalcule frontiere des grains ayant absorbe les pts grains
    for indg in listgrains:
        listpixfr = []
        for pix in TabGrD[indg-1]:
            lv = Lvoisinspix4(pix)
            for pixv in lv:
                if Tabrecsec2[pixv]!= indg: listpixfr.append(pixv)               
        TabGrD_vois[indg-1] = listpixfr
        
    print "End dilatation of the small grains"      

    #**********************************************************************************************

    EtatRun.set("\n"+"Just with Eulers. Nb of Grains = " + str(len(TabGrD)) + "Please wait..."+"\n")
    LabelRun["fg"] = "red"

    fctkicalgrain(TabGrD, TabGrD_vois)

    OptShowB = "euler1"

    ClassGrains()

    #if CrystD.syst == "Cubic" and SourceFile != 'TSL': IgnoreMisorientedGrainsDRemain() #LALA
    if CrystD.syst == "Cubic" : IgnoreMisorientedGrainsDRemain()
    
    IgnoreSmallGrainsD()

    CalculateGrainBoundaries("graind") # calcule les pixels qui sont en frontiere de grain

    Tabrecsec2 = []
    print
    print "the calculations lasted", (time.clock() - TimeStart) / 60, "min"
    print
    OptShowB = "graind"
    OptShowA = "euler1"
    refreshButtons1()
    refreshButtons2()
    ButtonM["foreground"] = "white"
    ButtonM["background"] = "blue"
    ButtonA["foreground"] = "white"
    ButtonA["background"] = "blue"

    if not HideFlag.get(): showmap()
    EtatRun.set("\n"+"Nb of Grains = " + str(len(TabGrains)) + ". Next step: Reconstruct the Parents Grains"+"\n")
    LabelRun["fg"] = "blue"
    UpDateProgress(0)
    frameFile.update()
    NextStep = 5
##    try: winsound.PlaySound("!", winsound.SND_ALIAS)
##    except: pass
    print "End of Daughter Reconstruction"

    if Transition != "Sigma3": IdentifyRetainedParentGrains()

#==========================================================================================================

def IgnoreSmallGrainsD():
    
    global OptShowB, TabParentGrains, TabGrainsRec, TabOrient, listNuclei, iparent
    global NextStep, TabPix
    
    TabParentGrains = []
    TabGrainsRec = []
    TabOrient = []
    listNuclei = []
    iparent = 0
    OptShowB = "graind"
    NextStep += 1
    IgnoreSmallGrains()
    
#==========================================================================================================

def IgnoreSmallGrainsP():
    
    global OptShowB
    
    OptShowB = "grainp"
    IgnoreSmallGrains()
    
#==========================================================================================================

def IgnoreSmallGrains():
    
    # retire les pts grains filles ou parents
    # attention : boucle iterative , pas de patch correctif
    
    global TabGrains, TabParentGrains, TabPix, OptShowB, OptShowA
    global canvas

    print "Ignoring Small Grains..."
    #os.chdir (os.path.join(EBSDDirectory, "Results"))

    UpDateProgress(0)
    frameFile.update()

    if OptShowB == "grainp":
        TabG = TabParentGrains
        tdp = 4
    else:
        OptShowB = "graind"
        TabG = TabGrains
        tdp = 3
    LL = len(TabG)

    print "NbGrains before ignoring = ",LL
    EtatRun.set("\n"+"Nb of grains before = " + str(len(TabG)) + ". Ignoring Small Grains..."+"\n")
    LabelRun["fg"] = "red"

    stepg = max(LL/StProg,1)
    #-------------------------------------------------------------------------------------------------
    # identification des pts grains entoures d au moins un gros grain
    TabGrainsToRem = []
    for indgr in range(1,LL+1):
        if not indgr%stepg: UpDateProgress((indgr * 100) / LL)
        gr = TabG[indgr-1]
        if len(gr.listpix) < MinGrainSize: # si petit grain
            sizegros,indvoisgros = 0, 0
            for indgrvois in gr.listgrvois: # regarde les grains voisins
                l = len(TabG[indgrvois - 1].listpix)
                if l > sizegros and l > MinGrainSize:  
                    sizegros = l
                    indvoisgros = indgrvois # prend le plus gros grain voisin
            if sizegros: TabGrainsToRem.append([indgr,indvoisgros]) #en 0 = to remove, en 1 = will remplace it
            else: # pas de gros grains voisins, va chercher alors dans les voisins des voisins
                indbis = 0
                for indgrvois in gr.listgrvois:
                    for indgrvoisvois in TabG[indgrvois-1].listgrvois:
                        if len(TabG[indgrvoisvois-1].listpix) > MinGrainSize: # prend le premier gros dans les seconds voisins
                            indbis = indgrvoisvois
                            break
                    if indbis: break
                if indbis: TabGrainsToRem.append([indgr,indbis])
                    
    #----------------------------------------------------------------
    # fusion des pts grains dans les gros voisins
    stepg = max(len(TabGrainsToRem)/StProg,1)
    ii=0
    for cind in TabGrainsToRem:
        if not ii%stepg: UpDateProgress((ii * 100) / len(TabGrainsToRem))
        indr,indn = cind 
        gr_cn = TabG[indn-1]
        gr_cr = TabG[indr-1]
        for indpix in gr_cr.listpix: # remplacement des pixels des petits grains
            pix = TabPix[indpix]
            pix[0], pix[1], pix[2] = gr_cn.euler1, gr_cn.euler2, gr_cn.euler3
            pix[tdp] = indn
        gr_cn.listpix.extend(gr_cr.listpix)
        for voiscr in gr_cr.listgrvois:
            if voiscr != indn and voiscr not in gr_cn.listgrvois: gr_cn.listgrvois.append(voiscr)
        if OptShowB == "grainp":
            gr_cn.graind.extend(gr_cr.graind)
        TabG[indr-1]=0
        ii+=1
    # renumerotation des grains   
    ii = 0
    if TabGrainsToRem:
        if not ii%stepg: UpDateProgress((ii * 100) / len(TabGrainsToRem))
        listcr = [x[0] for x in TabGrainsToRem]
        listcr.sort()
        while 0 in TabG: TabG.remove(0) # enleve juste les pts grains a pb, mais il faut renumeroter            
        if OptShowB == "graind":
            for grd in TabG:
                grd.graind = renum(grd.graind,listcr)
                grd.listgrvois = [renum(x,listcr) for x in grd.listgrvois]
                while 0 in grd.listgrvois: grd.listgrvois.remove(0)
            for pix in TabPix: pix[3]= renum(pix[3],listcr)
        elif OptShowB == "grainp":
            for grp in TabG:
                grp.grainp = renum(grp.grainp,listcr)
                grp.listgrvois = [renum(x,listcr) for x in grp.listgrvois]
                while 0 in grp.listgrvois: grp.listgrvois.remove(0)
            for pix in TabPix: pix[4]= renum(pix[4],listcr)
        ii+=1

        TabGrainsToRem = []
        listcr = []
        lrenum = []
        UpDateProgress(0)
        IgnoreSmallGrains()
        
    else:  
        if OptShowB == "graind": CalculateGrainBoundaries("graind")
        elif OptShowB == "grainp":
            TransfTabParentTabRec()
            CalculateGrainBoundaries("grainp")
        else: print "choose daughter or parent grain mode"

        refreshButtons1()
        refreshButtons2()
        

        if OptShowB == "graind":
            OptShowA = "euler1"
            ButtonM["foreground"] = "white"
            ButtonM["background"] = "blue"
            ButtonA["foreground"] = "white"
            ButtonA["background"] = "blue"

        elif OptShowB == "grainp":
            OptShowA = "numbers"
            ButtonN["foreground"] = "white"
            ButtonN["background"] = "blue"
            ButtonE["foreground"] = "white"
            ButtonE["background"] = "blue"

        if not HideFlag.get(): showmap()

        UpDateProgress(0)

        if OptShowB == "graind":
            EtatRun.set("\n"+"After: Nb grains = " + str(len(TabGrains)) + " NextStep: Reconstruct the Parent Grains."+"\n")
        elif OptShowB == "grainp":
            EtatRun.set("\n"+"After: Nb grains = " + str(len(TabParentGrains)) + "."+"\n")
        LabelRun["fg"] = "blue"

        frameFile.update()

 #==========================================================================================================

def IgnoreMisorientedGrainsDRemain():
    
    global OptShowB, TabParentGrains, TabGrainsRec, TabOrient, listNuclei, iparent
    
    TabParentGrains, TabGrainsRec,TabOrient, listNuclei = [], [], [] ,[]
    iparent = 0
    OptShowB = "graind"
    remain = 1
    IgnoreMisorientedGrains(-1, remain) # introduit interation avril 2012

#==========================================================================================================

def IgnoreMisorientedGrainsD():
    
    global OptShowB, TabParentGrains, TabGrainsRec, TabOrient, listNuclei, iparent
    
    TabParentGrains, TabGrainsRec,TabOrient, listNuclei = [], [], [] ,[]
    iparent = 0
    OptShowB = "graind"
    IgnoreMisorientedGrains(1)

#==========================================================================================================

def IgnoreMisorientedGrainsDIterate():
    global OptShowB
    OptShowB = "graind"
    IgnoreMisorientedGrains(-1)

#==========================================================================================================

def IgnoreMisorientedGrainsP():
    global OptShowB
    global NextStep
    OptShowB = "grainp"
    IgnoreMisorientedGrains(1)
    NextStep += 1

#==========================================================================================================

def IgnoreMisorientedGrainsPSigma3():
    global OptShowB
    global NextStep

    info = "No possibility to come back. Save the arpge file before this operation."
    tkMessageBox.showinfo("Warning", info)     
    OptShowB = "grainp"
    IgnoreMisorientedGrains(1,0,1)
    NextStep += 1
    
#==========================================================================================================

def IgnoreMisorientedGrainsPIterate():
    global OptShowB
    OptShowB = "grainp"
    IgnoreMisorientedGrains(-1)

#==========================================================================================================

def IgnoreMisorientedGrains(nstep= -1, remain=0, S3=0):
    
    #enleve les petits grains filles, a faire avant la reconstruction
    # nbstep = -1 pour iteration infinie, cad tant qu il le faut
    global TabGrains, TabParentGrains, TabPix, OptShowB, OptShowA
    global TabG, canvas

    os.chdir (os.path.join(EBSDDirectory, "Results"))

    if OptShowB == "graind":
        TabG = TabGrains
        Group = GroupeRotD
        TM = TMD
        TS = TSD

    elif OptShowB == "grainp":
        TabG = TabParentGrains
        Group = GroupeRotP
        TM = TMP
        TS = TSP

    if remain:Group = FEulersD[1]

    EtatRun.set("\n"+"Nb of Grains before = " + str(len(TabG)) + ". Disregarding Low Misorientations..."+"\n")
    LabelRun["fg"] = "red"
    frameFile.update()
    #SaveOldResults() 
    TabGrainsNew = []

    #--------------------------
    def OneRunUnify():
        global TabG, TabPix, TabGrains

        EtatRun.set("\n"+"Nb of Grains = " + str(len(TabG)) + ". Disregarding Low Misorientations..."+"\n")
        LabelRun["fg"] = "red"

        flag = 0
        # 1er passage pour detecter les grans qui vont ensemble
        TabInd = [0 for i in range(len(TabG))]
        i = 1
        k = 1
        stepg = max(len(TabG)/StProg,1)
        # -----low misorientations-------"
        if not S3:
            for gr in TabG:
                if not i%stepg: UpDateProgress((i * 100) / len(TabG))
                if not TabInd[i - 1]:
                    TabInd[i - 1] = k
                    for indgrvois in gr.listgrvois:
                        if not TabInd[indgrvois - 1]:
                            grvois = TabG[indgrvois - 1]
                            if grvois.euler1 != 0: # euler1 = 0 pile veut dire quil na pas reussi a trouve lorientation du grain
                                if testId(disorientationgrain(i, indgrvois, OptShowB), AngTolGrain):
                                    TabInd[indgrvois - 1] = k
                                    flag = 1
                    k = k + 1
                i = i + 1
        # ---------------Sigma3 ---------"
        else: 
            for gr in TabG:
                if not i%stepg: UpDateProgress((i * 100) / len(TabG))
                if not TabInd[i - 1]:
                    TabInd[i - 1] = k
                    for indgrvois in gr.listgrvois:
                        if not TabInd[indgrvois - 1]:
                            grvois = TabG[indgrvois - 1]
                            if grvois.euler1 != 0: # euler1 = 0 pile veut dire quil na pas reussi a trouve lorientation du grain
                                if testS3(disorientationgrain(i, indgrvois, OptShowB), AngTolGrain):
                                    TabInd[indgrvois - 1] = k
                                    flag = 1
                    k = k + 1
                i = i + 1
        #--------------------------------"
        if flag:
            TabInd2 = copy.deepcopy(TabInd)
            # 2eme passage pour refaire les liste de grains bien numerotees
            kmax = k
            TabIndToUnify = []
            for k in range(1, kmax):
                lprov = []
                while k in TabInd2:
                    i = TabInd2.index(k)
                    lprov.append(i + 1)
                    TabInd2[i] = 0
                TabIndToUnify.append(lprov)

            k = 1
            newTabG = []
            for listindgr in TabIndToUnify:
                grU = cloneGrain(TabG[listindgr[0] - 1])
                if OptShowB == "grainp":
                    grU.grainp = k
                    listgrd = grU.graind
                    for indgrd in listgrd : TabGrains[indgrd - 1].grainp = k
                else:
                    grU.graind = k
                listindgrvois = [TabInd[indp - 1] for indp in grU.listgrvois]
                for ind in listindgrvois:
                    while listindgrvois.count(ind) > 1: listindgrvois.pop(listindgrvois.index(ind))
                if k in listindgrvois: listindgrvois.pop(listindgrvois.index(k))

                newlistpix = grU.listpix
                newUxy = dot(len(newlistpix), array([grU.x, grU.y]))
                for indgr in listindgr[1:]:
                    grtoU = TabG[indgr - 1]
                    newUxy = newUxy + dot(len(grtoU.listpix), array([grtoU.x, grtoU.y]))
                    if OptShowB == "grainp":
                        listgrd = grtoU.graind
                        for indgrd in listgrd :
                            TabGrains[indgrd - 1].grainp = k
                            TabGrains[indgrd - 1].variant = [] #because the original parent grain was different
                        grU.graind.extend(listgrd) #30/10/08
                    #unification des grains
                    newlistpix.extend(grtoU.listpix)#attribue les pixels des grains voisins
                    for indtoU in grtoU.listgrvois: #calcule les grains voisins
                        indtoUref = TabInd[indtoU - 1]
                        if indtoUref != k and indtoUref != 0 and indtoUref not in listindgrvois:
                            listindgrvois.append(indtoUref)
                grU.listgrvois = listindgrvois
                grU.listpix = newlistpix
                grU.x, grU.y = dot(1. / len(newlistpix), newUxy)
                if OptShowB == "grainp":
                    for indpix in grU.listpix: TabPix[indpix][4] = k
                else:
                    for indpix in grU.listpix: TabPix[indpix][3] = k
                newTabG.append(grU)
                k = k + 1
            TabG = newTabG
            #print "lentabG",len(newTabG)
            return 1
        else:
            return 0

  #------------------------------------

    while OneRunUnify():
        if nstep:break
        pass

    if OptShowB == "graind": TabGrains = TabG
    elif OptShowB == "grainp":
        TabParentGrains = TabG
        TransfTabParentTabRec()

    #------------------------------------

    #SaveNewResults()

    CalculateGrainBoundaries(OptShowB)

    
    refreshButtons1()
    refreshButtons2()
    
    if OptShowB == "grainD":
        OptShowA = "euler1"
        ButtonM["foreground"] = "white"
        ButtonM["background"] = "blue"
        ButtonA["foreground"] = "white"
        ButtonA["background"] = "blue"
    elif OptShowB == "grainp":
        OptShowA = "numbers"
        ButtonN["foreground"] = "white"
        ButtonN["background"] = "blue"
        ButtonE["foreground"] = "white"
        ButtonE["background"] = "blue"

    if not HideFlag.get(): showmap()

    EtatRun.set("\n"+"After. Nb of Grains = " + str(len(TabG))+"\n")
    LabelRun["fg"] = "blue"
    frameFile.update()

    TabG = []

    UpDateProgress(0)

#==========================================================================================================

def Lvoisinspix4(x):

    if GRID == "Square":
        res = 0
        if x == 0: res = [1, Nbpixx]
        elif (0 < x <= (Nbpixx - 2)): res = [x - 1, x + 1, x + Nbpixx]
        elif x == Nbpixx -1: res = [x - 1, x + Nbpixx]
        elif (Nbpixx <= x <= (Longueur - Nbpixx - 1)):
            if (x + 1) % Nbpixx == 0: res = [x - Nbpixx, x - 1, x + Nbpixx]
            else:
                if x % Nbpixx == 0: res = [x - Nbpixx, x + 1, x + Nbpixx]
                else: res = [x - Nbpixx, x - 1, x + 1, x + Nbpixx]
        elif x == Longueur - Nbpixx: res = [x -  Nbpixx, x + 1]
        elif Longueur - Nbpixx + 1 <= x <= Longueur - 2: res = [x - Nbpixx, x - 1, x + 1]
        elif x == Longueur - 1: res = [x - Nbpixx, x - 1]
        
    elif GRID == "Hexagon": 
        res = 0
        if x == 0: res = [1, Nbpixx]
        elif (0 < x <= (Nbpixx - 1)): res = [x - 1, x + 1, x + Nbpixx,x + Nbpixx+1]
        elif (Nbpixx <= x <= (Longueur - Nbpixx - 1)):
            if (x + 1) % Nbpixx == 0: res = [x - Nbpixx, x - 1, x + Nbpixx]
            else:
                if x % Nbpixx == 0: res = [x - Nbpixx, x + 1, x + Nbpixx]
                else:
                    if (x/Nbpixx) %2 == 1: res = [x - Nbpixx, x - Nbpixx + 1, x - 1, x + 1, x + Nbpixx,x + Nbpixx+1]
                    else: res = [x - Nbpixx-1, x - Nbpixx, x - 1, x + 1, x + Nbpixx -1, x + Nbpixx]
        elif x == Longueur - Nbpixx: res = [Longueur - 2 * Nbpixx, Longueur - Nbpixx + 1]
        elif Longueur - Nbpixx + 1 <= x <= Longueur - 2: res = [x - Nbpixx, x - 1, x + 1]
        elif x == Longueur - 1: res = [Longueur - 1 - Nbpixx, Longueur - 1]

    return res

#==========================================================================================================

def Lvoisinspix8(x):

    res = 0

    if x == 0: # coin haut gauche
        res = [1, Nbpixx,Nbpixx+1]
    elif x == Longueur-Nbpixx: # coin bas gauche
        res = [x-Nbpixx, x-Nbpixx+1,x+1]
    elif x == Nbpixx-1: # coin haut droite
        res = [x-1, x+Nbpixx-1, x+Nbpixx]
    elif x == Longueur-1: # coin bas droite
        res = [x-1, x-Nbpixx-1, x-Nbpixx]
    elif (0 < x < (Nbpixx - 1)): # ligne haut
        res = [x-1, x+1, x+Nbpixx-1, x+Nbpixx, x+Nbpixx+1]
    elif (Longueur-Nbpixx < x < Longueur - 1): # ligne bas
        res = [x-1, x+1, x-Nbpixx-1, x-Nbpixx, x-Nbpixx+1]
    elif not x%Nbpixx : # colonne gauche
        res = [x-Nbpixx, x+Nbpixx, x-Nbpixx+1, x+1, x+Nbpixx+1]
    elif not (x+1)%Nbpixx : # colonne droite
        res = [x-Nbpixx, x+Nbpixx, x-Nbpixx-1, x-1, x+Nbpixx-1]   
    else: # cas general
        res = [x-Nbpixx-1, x-Nbpixx, x-Nbpixx+1, x-1, x+1, x+Nbpixx-1, x+Nbpixx, x+Nbpixx+1]

    return res

#====================================================================================================

def NextpixBin2(x):
    
    if x == 0: res = 2
    elif ((x+1) % Nbpixx) <= (Nbpixx - 2): res = x + 2
    else : res = x + 2 + Nbpixx
    if res >= Longueur: res = 0

    return res

#==========================================================================================================

def disorientationpixel(x, y):

    m = dot(EulersToMat(TabPix[x][0], TabPix[x][1], TabPix[x][2],SourceFile),
          inv(EulersToMat(TabPix[y][0], TabPix[y][1], TabPix[y][2],SourceFile)))
    mcryst = dot(invTSD, dot(m, TSD))
    mdis = Disorientation(mcryst, GroupeRotD)

    return mdis

#====================================================================================================

def disorientationpixelP(x, y):

    m = dot(EulersToMat(TabPixP[x][0], TabPixP[x][1], TabPixP[x][2],SourceFile),
          inv(EulersToMat(TabPixP[y][0], TabPixP[y][1], TabPixP[y][2],SourceFile)))
    mcryst = dot(invTSP, dot(m, TSP))
    mdis = Disorientation(mcryst, GroupeRotP)

    return mdis

#====================================================================================================

def disorientationpixelANGAXE(x, y):

    anglemin = 360

    m = dot((EulersToMat(TabPix[x][0], TabPix[x][1], TabPix[x][2],SourceFile)),
          inv(EulersToMat(TabPix[y][0], TabPix[y][1], TabPix[y][2],SourceFile)))

    mcryst = dot(invTSD, dot(m, TSD))
    mdis = Disorientation(mcryst, GroupeRotD)

    angle, axe = calculeRot(mdis)
    
    return [angle, axe]     


#====================================================================================================

def disorientationpixelANGAXEP(x, y):

    result = []
    anglemin = 360

    m = dot((EulersToMat(TabPixP[x][0], TabPixP[x][1], TabPixP[x][2],SourceFile)),
          inv(EulersToMat(TabPixP[y][0], TabPixP[y][1], TabPixP[y][2],SourceFile)))

    mcryst = dot(invTSP, dot(m, TSP))
    mdis = Disorientation(mcryst, GroupeRotP)

    angle, axe = calculeRot(mdis)
##    if CrystP.syst == "Cubic": angle, axe = calculeRot(mdis)
##    else:
##        mkortho = dot(TSP,dot(mdis,inv(TSP))) # to transform it into ORTHO to ORTHO
##        rot = calculeRot(mkortho)
##        angle, axe = [rot[0],IntegriseAxe(dot(inv(TP),rot[1]))]

    return [angle, axe]


#====================================================================================================

def disorientationgrain(pi,pj, opt="graind"):

    x,y = pi-1,pj-1

    if OptShowB == "graind" or opt == "graind":
        Group = GroupeRotD
        TM = TMD
        TS = TSD
        syst = CrystD.syst
        TabG = TabGrains
        m = dot(EulersToMat(TabG[x].euler1, TabG[x].euler2, TabG[x].euler3, SourceFile),
              inv(EulersToMat(TabG[y].euler1, TabG[y].euler2, TabG[y].euler3, SourceFile)))
        
    elif (OptShowB == "grainp" or opt == "grainp") and (pi>0 and pj>0):
        Group = GroupeRotP
        TM = TMP
        TS = TSP
        syst = CrystP.syst
        TabG = TabParentGrains
        m = dot(EulersToMat(TabG[x].euler1, TabG[x].euler2, TabG[x].euler3, SourceFile),
              inv(EulersToMat(TabG[y].euler1, TabG[y].euler2, TabG[y].euler3, SourceFile)))
        
    else:
        Group = GroupeRotP
        TM = TMP
        TS = TSP
        syst = CrystP.syst
        if pi>0: TabGi = TabParentGrains
        else:
            TabGi = TabRetainedGrains
            x = abs(pi)-1
        if pj>0: TabGj = TabParentGrains
        else:
            TabGj = TabRetainedGrains
            y = abs(pj)-1
        m = dot(EulersToMat(TabGi[x].euler1, TabGi[x].euler2, TabGi[x].euler3, SourceFile),
              inv(EulersToMat(TabGj[y].euler1, TabGj[y].euler2, TabGj[y].euler3, SourceFile)))

    mcryst = dot(inv(TS), dot(m, TS))
    mdis = Disorientation(mcryst, Group)

    return mdis

#====================================================================================================

def disorientationgrainANGAXE(i, j,opt = "graind"):

    x,y = i-1, j-1

    if opt == "graind":
        Group = GroupeRotD
        TM = TMD
        TS = TSD
        syst = CrystD.syst
        TabG = TabGrains
        m = dot(EulersToMat(TabG[x].euler1, TabG[x].euler2, TabG[x].euler3, SourceFile),
              inv(EulersToMat(TabG[y].euler1, TabG[y].euler2, TabG[y].euler3, SourceFile)))
        
    elif opt == "grainp" and i>1 and j>1:
        Group = GroupeRotP
        TM = TMP
        TS = TSP
        syst = CrystP.syst
        TabG = TabParentGrains
        m = dot(EulersToMat(TabG[x].euler1, TabG[x].euler2, TabG[x].euler3, SourceFile),
              inv(EulersToMat(TabG[y].euler1, TabG[y].euler2, TabG[y].euler3, SourceFile)))
        
    else:
        Group = GroupeRotP
        TM = TMP
        TS = TSP
        syst = CrystP.syst
        if i>0: TabGi = TabParentGrains
        else:
            TabGi = TabRetainedGrains
            x = abs(i)-1
        if j>0: TabGj = TabParentGrains
        else:
            TabGj = TabRetainedGrains
            y = abs(j)-1
        m = dot(EulersToMat(TabGi[x].euler1, TabGi[x].euler2, TabGi[x].euler3, SourceFile),
              inv(EulersToMat(TabGj[y].euler1, TabGj[y].euler2, TabGj[y].euler3, SourceFile)))

    mcryst = dot(inv(TS), dot(m, TS))
    mdis = Disorientation(mcryst, Group)
     
    if syst == "Cubic":
        angle, axe = calculeRot(mdis)
    else:
        mkortho = dot(TS,dot(mdis,inv(TS))) # to transform it into ORTHO to ORTHO
        rot = calculeRot(mkortho)
        angle, axe = [rot[0],IntegriseAxe(dot(inv(TS),rot[1]))]

    return [angle, axe]

#====================================================================================================
#oct2017
def GrainNucleation2(istart, angtol):

    global TabGrains
    
    finalres = []
    listres = []
    res = []
    #print "***************************************"
    #print "nucleation step",istart,"angtol=",angtol
    stepg = max(len(TabGrains)/StProg,1)
    for i in range(istart + 1, len(TabGrains) + 1):
        if not i%stepg: UpDateProgress((i * 100) / len(TabGrains))
        finalres = []
        listres = []
        gr = TabGrains[i - 1]
        res = []
        if not gr.variant:
            listcand = [] # ceux qui nappartiennent pas deja a un grain parent
            listindvois = gr.listgrvois
            listindvois.sort() #tri les voisins par taille avec les gros en premier
            for j in listindvois:
                grj = TabGrains[j - 1]
                if not grj.variant: listcand.append(j)
            if len(listcand) >= 2:
                listcand2 = [] # ceux qui sont lies au grain i par un operateur different de lidentite
                for j in listcand:
                    t = test(disorientationgrain(i, j), angtol)
                    if t and t != [0]:
                        listcand2.append([j, t])
                if len(listcand2) >= 1:
                    j = listcand2[0][0]
                    listdesij = listcand2[0][1]
                    res = [[i, j], listdesij[0]] # ne prend le res que si les op sont diff de l identite = op0
                    break
        #if len(listres) == 1: break # ne garde que si il y a qu 1 seule solution possible
    if res:
        i, j = res[0]
        TabGrains[i - 1].variant = [1] # par reference
        TabGrains[i - 1].grainp = iparent + 1
        TabGrains[i - 1].confind = angtol
        TabGrains[j - 1].variant = [1] # car methode par defaut sans possibilite dorienter
        TabGrains[j - 1].grainp = iparent + 1
        TabGrains[j - 1].confind = angtol

    else: pass

    return res

#====================================================================================================
#====================================================================================================

def GrainNucleation3(istart, angtol):

    global TabGrains
    
    # cherche un noyau de trois grains coherents a partir du ieme grain fille
    finalres = []
    listres = []
    res = []
    #print "***************************************"
    #print "nucleation step",istart,"angtol=",angtol
    stepg = max(len(TabGrains)/StProg,1)
    for i in range(istart + 1, len(TabGrains) + 1):
        if not i%stepg: UpDateProgress((i * 100) / len(TabGrains))
        finalres = []
        listres = []
        gr = TabGrains[i - 1]
        res = []
        if not gr.variant:
            listcand = [] # ceux qui nappartiennent pas deja a un grain parent
            listindvois = gr.listgrvois
            listindvois.sort() #tri les voisins par taille avec les gros en premier
            for j in listindvois:
                grj = TabGrains[j - 1]
                if not grj.variant: listcand.append(j)
            if len(listcand) >= 2:
                listcand2 = [] # ceux qui sont lies au grain i par un operateur different de lidentite
                for j in listcand:
                    t = test(disorientationgrain(i, j), angtol)
                    if t and t != [0]:
                        listcand2.append([j, t])
                if len(listcand2) >= 2:
                    j = listcand2[0][0]
                    listdesij = listcand2[0][1]
                    k = listcand2[1][0]
                    listdesik = listcand2[1][1]
                    listdesjk = test(disorientationgrain(j, k), angtol)
                    for dij in listdesij:
                        for dik in listdesik:
                            for djk in listdesjk:
                                if djk in TabCompoOp[dij][dik]: #ATTENTION VERIFIER QUE COMPO = OPi-1 OPj, a priori cest OK
                                    if dij and dik and djk:
                                        res = [[i, j, k], [dij, dik, djk]] # ne prend le res que si les op sont diff de l identite = op0
                                        listres.append(res)
                                        #break
                            #if res: break
                        #if res: break
        # mofifie en janv 2020 pour permettre reconst NiTi
        if Transition == "NiTi": cond = (len(listres) >= 1)
        else: cond = (len(listres) == 1)
        if cond: break
        
    if listres:
        res = listres[0]
        i, j, k = res[0]
        desij, desik, desjk = res[1]
        TabGrains[i - 1].variant = [1] # par reference
        TabGrains[i - 1].grainp = iparent + 1
        TabGrains[i - 1].confind = angtol
        TabGrains[j - 1].grainp = iparent + 1
        TabGrains[j - 1].confind = angtol
        TabGrains[k - 1].grainp = iparent + 1
        TabGrains[k - 1].confind = angtol
        #cherche maintenant une indexation coherente des variants => 1 seul numero pour les variants des noyaux
        lOpjk = compoOpVar1(desij, desik) # sous forme {[[s,t],op]}
        for Opjk in lOpjk:
            if Opjk[1] == desjk:
                if Opjk[0][0] not in TabGrains[j - 1].variant: TabGrains[j - 1].variant.append(Opjk[0][0])
                if Opjk[0][1] not in TabGrains[k - 1].variant: TabGrains[k - 1].variant.append(Opjk[0][1])
    else: pass

    return res

#====================================================================================================

def GrainNucleation4(istart, angtol):
    
    global TabGrains
    # cherche un noyau de 4 grains coherents a partir du ieme grain fille
    
    finalres = []
    listres = []
    #print "nucleation step",istart,"angtol=",angtol
    stepg = max(len(TabGrains),1)
    for i in range(istart + 1, len(TabGrains) + 1):
        if not i%stepg: UpDateProgress((i * 100) / len(TabGrains))
        finalres = []
        listres = []
        gr = TabGrains[i - 1]
        if not gr.variant:
            listcand = [] # ceux qui nappartiennent pas deja a un grain parent
            listindvois = gr.listgrvois
            listindvois.sort() #tri les voisins par taille avec les gros en premier
            for j in listindvois:
                grj = TabGrains[j - 1]
                if not grj.variant: listcand.append(j)
            if len(listcand) >= 3:
                listcand2 = [] # ceux qui sont lies au grain i par un operateur different de lidentite
                for j in listcand:
                    t = test(disorientationgrain(i, j), angtol)
                    if t and t != [0]:
                        listcand2.append([j, t])
                if len(listcand2) >= 3:
                    j = listcand2[0][0]
                    listdesij = listcand2[0][1]
                    k = listcand2[1][0]
                    listdesik = listcand2[1][1]
                    l = listcand2[2][0]
                    listdesil = listcand2[2][1]
                    listdesjk = test(disorientationgrain(j, k), angtol)
                    listdesjl = test(disorientationgrain(j, l), angtol)
                    listdeskl = test(disorientationgrain(k, l), angtol)
                    lres1, lres2, lres3 = [], [], []
                    #------------------------------------------------------------------------------
                    if listdesjk and listdesjl and listdeskl:

                        for dij in listdesij:
                            for dik in listdesik:
                                for djk in listdesjk:
                                    if djk in TabCompoOp[dij][dik]: #ATTENTION VERIFIER QUE COMPO = OPi-1 OPj, a priori cest OK
                                        if dij and dik and djk:
                                            res1 = [dij, dik, djk] # ne prend le res que si les op sont diff de l identite = op0
                                            lres1.append(res1)
    ##                                        break
    ##                            if res1: break
    ##                        if res1: break

                        for dij in listdesij:
                            for dil in listdesil:
                                for djl in listdesjl:
                                    if djl in TabCompoOp[dij][dil]:
                                        if dij and dil and djl:
                                            res2 = [dij, dil, djl]
                                            lres2.append(res2)
    ##                                        break
    ##                            if res2: break
    ##                        if res2: break

                        for dik in listdesik:
                            for dil in listdesil:
                                for dkl in listdeskl:
                                    if dkl in TabCompoOp[dik][dil]:
                                        if dik and dil and dkl:
                                            res3 = [dik, dil, dkl]
                                            lres3.append(res3)
    ##                                        break
    ##                            if res3: break
    ##                        if res3: break


    ##                    for djk in listdesjk: 
    ##                        for djl in listdesjl:
    ##                            for dkl in listdeskl:
    ##                                if dkl in TabCompoOp[djk][djl]: #ne marche pas car la variant j nest pas 1
    ##                                    if djk and djl and dkl:
    ##                                        res4=[djk,djl,dkl]
    ##                                        print "res4",res4
    ##                                        break
    ##                            if res4: break
    ##                        if res4: break
                      #------------------------------------------------------------------------------                  
                    #print lres1,lres2,lres3
                    if lres1 and lres2 and lres3:
                        #print "OK, good", lres1,lres2,lres3
                        for res1 in lres1:
                            for res2 in lres2:
                                for res3 in lres3:
                                    if (res1[0] == res2[0] and res1[1] == res3[0] and res2[1] == res3[1]):
                                        dij, dik, djk = res1
                                        dil, djl = res2[1], res2[2]
                                        dkl = res3[2]
                                        #print "ok growth",[i,j,k,l],[dij,dik,dil,djk,djl,dkl]
                                        if [[i, j, k, l], [dij, dik, dil, djk, djl, dkl]] not in listres: listres.append([[i, j, k, l], [dij, dik, dil, djk, djl, dkl]])
                                        break
                                if listres: break
                            if listres: break

        if listres: break
    if listres: # ne garde que la 1ere des solutions possibles, mais en general il ny en a qu une dapres mes tests
        finalres = listres[0]
        res = listres[0]
        i, j, k, l = res[0]
        desij, desik, desil, desjk, desjl, deskl = res[1]
        #print "desij,desik,desil,desjk,desjl,deskl",desij,desik,desil,desjk,desjl,deskl
        TabGrains[i - 1].variant = [1] # par reference
        TabGrains[i - 1].grainp = iparent + 1
        TabGrains[i - 1].confind = angtol
        TabGrains[j - 1].grainp = iparent + 1
        TabGrains[j - 1].confind = angtol
        TabGrains[k - 1].grainp = iparent + 1
        TabGrains[k - 1].confind = angtol
        TabGrains[l - 1].grainp = iparent + 1
        TabGrains[l - 1].confind = angtol
        #cherche maintenant une indexation coherente des variants => 1 seul numero pour les variants des noyaux
        lOpjk = compoOpVar1(desij, desik) # sous forme {[[s,t],op]}
        for Opjk in lOpjk:
            if Opjk[1] == desjk:
                if Opjk[0][0] not in TabGrains[j - 1].variant:
                    #print "jk","j=",Opjk[0][0]
                    TabGrains[j - 1].variant.append(Opjk[0][0])
                if Opjk[0][1] not in TabGrains[k - 1].variant:
                    #print "jk","k=",Opjk[0][1]
                    TabGrains[k - 1].variant.append(Opjk[0][1])
                break
        lOpjl = compoOpVar1(desij, desil) # sous forme {[[s,t],op]}
        for Opjl in lOpjl:
            if Opjl[1] == desjl:
                if Opjl[0][0] not in TabGrains[j - 1].variant:
                    #print "jl","j=",Opjl[0][0]
                    TabGrains[j - 1].variant.append(Opjl[0][0])
                if Opjl[0][1] not in TabGrains[l - 1].variant:
                    #print "jl","l=",Opjl[0][1]
                    TabGrains[l - 1].variant.append(Opjl[0][1])
                break

    return finalres

#====================================================================================================

def ReconstParentGrainsOrient2 ():
    ReconstParentGrainsOrient (2)
    
#====================================================================================================

def ReconstParentGrainsOrient3 ():
    ReconstParentGrainsOrient (3)

#====================================================================================================

def ReconstParentGrainsOrient4 ():
    ReconstParentGrainsOrient (4)

#====================================================================================================

def ReconstParentGrainsOrient (nucleus=3):
    
    # nucleus = 3 for triplets and 4 for quadruplets
    global TabPix, TabGrains, TabParentGrains
    global Transition, Nbpixx, Nbpixy, XStep, Xmax, Ymax
    global OptShowA, OptShowB, ComptNucl, ComptGrowth, iparent, FlagRec
    global TabGrainsRec, TabOrient, listIndGrdGrowth
    global NextStep, PercentRec, canvas

    if len(TabGrains)== 0:
        IdentifyDaughterGrains()
        
    if ComptNucl > 0:
        pass
        #SaveOldResults()

    if nucleus==3: EtatRun.set("\n"+"Nucleation + Growth1 with triplets. Running..."+"\n")
    elif nucleus==4: EtatRun.set("\n"+"Nucleation + Growth1 with quadruplets. Running..."+"\n")
    elif nucleus==2: EtatRun.set("\n"+"Nucleation + Growth1 with doublets. Running..."+"\n")
    LabelRun["fg"] = "red"
    frameFile.update()

    TimeStart = time.clock()

    if FlagRec == 1:
        print "Reseting----------"
        FlagRec = 0
        #pour remttre a zero les compteurs
        TabGrainsRec = []
        TabOrient = []
        iparent = 0
        for i in range(len(TabPix)): TabPix[i][4] = 0 #4 = grainp
        for gr in TabGrains:
            gr.grainp = 0
            gr.variant = []

    if ComptNucl == 0:
        ComptNucl+=1
        istart = 0
        flagNucl = 1
        if nucleus==3: print "Reconstruction with triplets. Running..."
        elif nucleus==4: print "Reconstruction  with quadruplets. Running..."
        elif nucleus==2: print "Reconstruction  with doublets. Running..."

        #print "----------------------NUCLEATION /FIRST GROWTH----------------------------"

        if nucleus==3:
            while flagNucl:
                res = GrainNucleation3(istart, AngTolParentNucl)
                if res:
                    i, j, k = res[0] # i = grain trouve ayant nuclee, j et k grains voisins en coherence avec i
                    dij, dik, djk = res[1] # operateurs de disorientation entre grains ij,ik,jk
                    listgrexpl = []
                    for ind in res[0]: #i,j,k
                        for indgrvois in TabGrains[ind - 1].listgrvois:
                            if not TabGrains[indgrvois - 1].variant:
                                if indgrvois not in listgrexpl: listgrexpl.append(indgrvois)
                    listIndGrdGrowth = [i,j,k]
                    GrainGrowth3(res, listgrexpl, AngTolParentGrowth / 2.)
                    eulers_grp = OneParentGrainOrientation(listIndGrdGrowth)
                    if eulers_grp[0]!=0 and len(listIndGrdGrowth)> NbMinGrd:
                        for indgrd in listIndGrdGrowth :
                            TabGrains[indgrd-1].grainp = iparent+1
                        TabGrainsRec.append(listIndGrdGrowth)       
                        TabOrient.append(eulers_grp)
                        iparent+= 1
                    else:
                        ResetListGrd(listIndGrdGrowth)
                    if i==len(TabGrains): break
                    for newistart in range(i, len(TabGrains)):
                        if not TabGrains[newistart].variant: break
                    istart = newistart+1
                    if istart == len(TabGrains): flagNucl = 0
                else:
                    flagNucl = 0

        elif nucleus==4:
            while flagNucl:
                res = GrainNucleation4(istart, AngTolParentNucl)
                if res:
                    i, j, k, l = res[0] # i = grain trouve ayant nuclee, j et k grains voisins en coherence avec i
                    dij, dik, dil, djk, djl, dkl = res[1] = res[1] # operateurs de disorientation entre grains ij,ik,jk
                    listgrexpl = []
                    for ind in res[0]: #i,j,k
                        for indgrvois in TabGrains[ind - 1].listgrvois:
                            if not TabGrains[indgrvois - 1].variant:
                                if indgrvois not in listgrexpl: listgrexpl.append(indgrvois)
                    listIndGrdGrowth = [i,j,k,l]
                    GrainGrowth4(res, listgrexpl, AngTolParentGrowth / 2.)
                    eulers_grp = OneParentGrainOrientation(listIndGrdGrowth)
                    if eulers_grp[0]!=0 and len(listIndGrdGrowth)> NbMinGrd:
                        for indgrd in listIndGrdGrowth :
                            TabGrains[indgrd-1].grainp = iparent+1
                        TabGrainsRec.append(listIndGrdGrowth)       
                        TabOrient.append(eulers_grp)
                        iparent+= 1
                    else:
                        ResetListGrd(listIndGrdGrowth)
                    if i==len(TabGrains): break
                    for newistart in range(i, len(TabGrains)):
                        if not TabGrains[newistart].variant: break
                    istart = newistart+1
                    if istart == len(TabGrains): flagNucl = 0
                else:
                    flagNucl = 0
               
        elif nucleus==2: #oct 2017
            while flagNucl:
                res = GrainNucleation2(istart, AngTolParentNucl)
                if res:
                    i, j = res[0] # i = grain trouve ayant nuclee, j et k grains voisins en coherence avec i
                    dij = res[1] # operateurs de disorientation entre grains ij,ik,jk
                    #print "line 6604, grain growth2",i,j,dij 
                    listgrexpl = []
                    for ind in res[0]: #i,j
                        for indgrvois in TabGrains[ind - 1].listgrvois:
                            if not TabGrains[indgrvois - 1].variant:
                                if indgrvois not in listgrexpl: listgrexpl.append(indgrvois)
                    listIndGrdGrowth = [i,j]
                    GrainGrowth2(res, listgrexpl, AngTolParentGrowth / 2.)
                    eulers_grp = OneParentGrainOrientation(listIndGrdGrowth,2)
                    #print " a voir", eulers_grp, len(listIndGrdGrowth)
                    if eulers_grp[0]!=0 and len(listIndGrdGrowth)> 2:
                        for indgrd in listIndGrdGrowth :
                            TabGrains[indgrd-1].grainp = iparent+1
                        TabGrainsRec.append(listIndGrdGrowth)       
                        TabOrient.append(eulers_grp)
                        iparent+= 1
                    else:
                        ResetListGrd(listIndGrdGrowth)
                    if i==len(TabGrains): break
                    for newistart in range(i, len(TabGrains)):
                        if not TabGrains[newistart].variant: break
                    istart = newistart+1
                    if istart == len(TabGrains): flagNucl = 0
                else:
                    flagNucl = 0

    UpDateProgress(0)
    #print "----------------------LAST GROWTH----------------------------"
    if nucleus==3: EtatRun.set("\n"+"Orientation + Growth2 with triplets. Running..."+"\n")
    elif nucleus==4: EtatRun.set("\n"+"Orientation + Growth2 with quadruplets. Running..."+"\n")
    elif nucleus==2: EtatRun.set("\n"+"Orientation + Growth2 with doublets. Running..."+"\n")
    LabelRun["fg"] = "red"
    frameFile.update()
    ComptGrowth+=1
    stepg = max(len(TabGrainsRec),1)
    for i in range(len(TabGrainsRec)):
        listindgrd = TabGrainsRec[i]
        try: eulers_grp = TabOrient[i]
        except: eulers_grp = OneParentGrainOrientation(listindgrd) #recalcul si on veut exploiter un fichier arpge juste ouvert
        PtoS = EulersToMat(eulers_grp[0],eulers_grp[1],eulers_grp[2],SourceFile) #S sample, P parent ortho frame
        StoP = dot(inv(PtoS), TSP) #P parent crystal
        listStoV = []
        for q in listQPos:
            StoV = dot(dot(StoP, q), PtoD)
            listStoV.append(StoV) #[matrice V1 to S,  matrice V2 to S ...], S = Sample, V = Variants  
        listgrexpl = []
        for ind in listindgrd: #i,j,k, et les autres grains ajoutes lors du graingrowth precedent
            for indgrvois in TabGrains[ind - 1].listgrvois:
                if not TabGrains[indgrvois - 1].variant and indgrvois not in listgrexpl: listgrexpl.append(indgrvois)
        GrainGrowthOrient(listStoV,listgrexpl, [],i, AngTolParentGrowth)
        if not i%stepg: UpDateProgress((i * 100) / len(TabGrainsRec))

    #***********************************************************************

    TabParentGrains = []
    for i in range(len(TabGrainsRec)):
        try: phi1,phi,phi2 = TabOrient[i]
        except:
            listindgrd = TabGrainsRec[i]
            phi1,phi,phi2 = OneParentGrainOrientation(listindgrd)
        grparent = cloneGrain(TabGrains[TabGrainsRec[i][0] - 1]) #a faire: recalculer les angles Euler pour le grain parent! et les liste voisins
        grparent.grainp = i + 1
        grparent.euler1 = phi1
        grparent.euler2 = phi
        grparent.euler3 = phi2
        grparent.graind = copy.deepcopy(TabGrainsRec[i])
        listvoisdaught = []
        #reattribut les voisins
        for j in range(len(TabGrainsRec[i])):
            lvois = TabGrains[TabGrainsRec[i][j] - 1].listgrvois
            for v in lvois:
                if v not in listvoisdaught: listvoisdaught.append(v)
        listvoispar = []
        for vd in listvoisdaught:
            for k in range(len(TabGrainsRec)):
                lgrp = TabGrainsRec[k]
                if vd in lgrp and k + 1 not in listvoispar and k != i: listvoispar.append(k + 1)
        #print "listvois =", listvoispar
        grparent.listgrvois = listvoispar
        xp, yp = [0, 0]
        lgraind = []
        grparent.listpix = []
        #reattribut les pixels
        for j in range(len(TabGrainsRec[i])):
            lgraind.append(TabGrainsRec[i][j])
            listpixij = TabGrains[TabGrainsRec[i][j] - 1].listpix
            grparent.listpix.extend(listpixij)
            TabGrains[TabGrainsRec[i][j] - 1].grainp = i + 1
            for kk in listpixij: TabPix[kk][4] = i + 1 #4 = grainp
            xp = xp + TabGrains[TabGrainsRec[i][j] - 1].x
            yp = yp + TabGrains[TabGrainsRec[i][j] - 1].y
        grparent.graind = lgraind
        grparent.x = xp / len(TabGrainsRec[i])
        grparent.y = yp / len(TabGrainsRec[i])
        TabParentGrains.append(grparent)

    CalculateGrainBoundaries("grainp")

    print "Cycle duration=", (time.clock() - TimeStart) / 60, "min"
    print "nb of parent grains", len(TabGrains)

    PercentRec = 0
    for grd in TabGrains:
        if grd.grainp:
            PercentRec += 1
    PercentRec = (100 * PercentRec) / len(TabGrains)

    ComptNucl+=1

    os.chdir (os.path.join(EBSDDirectory, "Results"))

    OptShowB = "grainp"
    OptShowA = "numbers"
    refreshButtons1()
    refreshButtons2()
    ButtonE["foreground"] = "white"
    ButtonE["background"] = "blue"
    ButtonN["foreground"] = "white"
    ButtonN["background"] = "blue"
    if not HideFlag.get(): showmap()

    EtatRun.set("\n"+"End " + str(PercentRec) + "% of grains reconstructed."+"\n")
    LabelRun["fg"] = "blue"
    UpDateProgress(0)
    NextStep += 1

#====================================================================================================

def TRDNucleation(istart, angtol,iparent):
    
    global TabGrains, TabGrainsRec
    # cherche un noyau de 2 grains coherents a partir du ieme grain fille

    #print "***************************************"
    #print "nucleation step",istart,"angtol=",angtol
    stepg = max(len(TabGrains)/StProg,1)
    res = []
    for i in range(istart + 1, len(TabGrains) + 1):
        if not i%stepg: UpDateProgress((i * 100) / len(TabGrains))
        gr = TabGrains[i - 1]
        if not gr.variant:
            listcand = [] # ceux qui nappartiennent pas deja a un grain parent
            listindvois = gr.listgrvois
            listindvois.sort() #tri les voisins par taille avec les gros en premier
            for j in listindvois:
                grj = TabGrains[j - 1]
                if not grj.variant: listcand.append(j)
            if len(listcand) >= 1:
                listcand2 = [] # ceux qui sont lies au grain i par un operateur different de lidentite
                for j in listcand:
                    t = testS3(disorientationgrain(i, j), angtol)
                    if t :
                        res = [i, j]
                        break
            if res: break
    if res:
        i, j = res
        TabGrains[i - 1].variant = [0]
        TabGrains[i - 1].grainp = iparent + 1
        TabGrains[i - 1].confind = angtol
        TabGrainsRec[iparent].append(i)
        TabGrains[j - 1].grainp = iparent + 1
        TabGrains[j - 1].confind = angtol
        TabGrainsRec[iparent].append(j)
        TabGrains[j - 1].variant = [0]
        #cherche maintenant une indexation coherente des variants => 1 seul numero pour les variants des noyaux

    return res

#====================================================================================================

def TRDFinish(angtol):
    
    global TabGrains, TabGrainsRec, iparent
    # ne sert qu'a mettre en grain parent tous les grains filles isoles cad sans sigma3

    res = []
    for i in range(len(TabGrains) + 1):
        gr = TabGrains[i - 1]
        if not gr.variant:
            TabGrains[i - 1].variant = [0]
            TabGrains[i - 1].grainp = iparent + 1
            TabGrains[i - 1].confind = angtol
            TabGrainsRec.append([])
            TabGrainsRec[iparent].append(i)
            iparent = iparent + 1
    return

#====================================================================================================

def GrainGrowthOrient(listStoV,listgrexplor,listProvGrExclus, iparent2, anggrowth):
    # fait grossir le triplet de grains filles i=var1,j,k
    # listgrexplor ne contient que des grains qui n appartiennent pas deja a un grain parent

    global TabGrains, TabGrainsRec, listvoisvar

    #print "listgrexplor",listgrexplor
    #liste de grains deja testes

    listvoisvar = [] #liste des voisins des variants qui seront trouves
    # calcul de ceux qui sont lies aux grains i, j et k par des operateurs coherent

    #---------------------------------------------------------------------------------------------
    def attr_varnum(m, indvar):
        global TabGrains, listvoisvar, TabGrainsRec
        TabGrains[m - 1].variant.append(indvar)
        TabGrains[m - 1].grainp = iparent2 + 1 # A VERIFIER
        TabGrains[m - 1].confind = anggrowth
        TabGrainsRec[iparent2].append(m) 
        listvoism = TabGrains[m - 1].listgrvois
        for voism in listvoism:
            if (len(TabGrains[voism - 1].variant) == 0) and (voism not in listvoisvar): listvoisvar.append(voism)
    #---------------------------------------------------------------------------------------------
    for m in listgrexplor:
        if m not in listProvGrExclus: listProvGrExclus.append(m)
        grd = TabGrains[m - 1]
        grd.variant = []
        DtoS = EulersToMat(grd.euler1, grd.euler2, grd.euler3,SourceFile) #S sample, grd daughter grains
        DtoS = dot(invTSD, DtoS)
        lang = []
        for iv in range(len(listStoV)):
            StoV = listStoV[iv]
            dtov = dot(DtoS, StoV)
            mdis = Disorientation(dtov, GroupeRotD)
            if mdis == []: mdis = dtov
            #print "det=",det(mdis)
            ang = testIdang(mdis)
            lang.append(ang)
        angmin = min(lang)
        if angmin<anggrowth:
            indvar = lang.index(angmin) + 1
            attr_varnum(m, indvar)

    newlistgrexplor = []
    for ind in listvoisvar: # ne garde dans la nouvelle list de gr a explorer que ceux qui n ont pas deja ete testes
        if ind not in listProvGrExclus: newlistgrexplor.append(ind)
    if newlistgrexplor:
##        print "--"
##        print newlistgrexplor
##        print listProvGrExclus
##        print listvoisvar
##        print "--"
##        print
        GrainGrowthOrient(listStoV,newlistgrexplor, listProvGrExclus,iparent2, anggrowth)
    else:
        return

#====================================================================================================
    #oct 2017

def GrainGrowth2(doublet, listgrexplor, anggrowth):

    global listIndGrdGrowth, TabGrains, listvoisvar
    
    # fait grossir le doublet de grains filles i=var1,j
    # listgrexplor ne contient que des grains qui n appartiennent pas deja a un grain parent
    
    #print "listgrexplor",listgrexplor
    #liste de grains deja testes

    listvoisvar = [] #liste des voisins des variants qui seront trouves
    # calcul de ceux qui sont lies aux grains i, j par des operateurs coherent

    #---------------------------------------------------------------------------------------------
    def attr_varnum(m, listvarm):
        global TabGrains, listvoisvar,listIndGrdGrowth
 
        for varm in listvarm:
            if varm not in TabGrains[m - 1].variant: TabGrains[m - 1].variant.append(varm)
        TabGrains[m - 1].confind = anggrowth
        listvoism = TabGrains[m - 1].listgrvois
        for voism in listvoism:
            if (len(TabGrains[voism - 1].variant) == 0) and (voism not in listvoisvar): listvoisvar.append(voism)

        if m not in listIndGrdGrowth: listIndGrdGrowth.append(m)
    #----------------------------------------------------

    try:
        i, j = doublet[0]
        dij = doublet[1]
    except:
        print "pb in GrainGrowth2"

    for m in listgrexplor:

        listdesmi = test(disorientationgrain(m, i), AngTolParentNucl) # si identite avec ang nucleation, plus facile
        listdesmj = test(disorientationgrain(m, j), AngTolParentNucl)

        if listdesmi == [0]: attr_varnum(m, TabGrains[i - 1].variant)
        elif listdesmj == [0]: attr_varnum(m, TabGrains[j - 1].variant)
            
        else:
            listdesmi = test(disorientationgrain(m, i), anggrowth)
            listdesmj = test(disorientationgrain(m, j), anggrowth)

            if listdesmi and listdesmj:

                listOpmi = []
                for dmi in listdesmi:
                    for dmj in listdesmj:
                        if dij in TabCompoOp[dmi][dmj]: #coherence i,j,m
                            if dmi not in listOpmi: listOpmi.append(dmi)

                if listOpmi:    #repasse sur les grains i,j,k mais cette fois avec une precision sur les dmi potentiels
                                # va donc calculer les numeros des variants du grain m en fonction des numeros de variants des grains i,j,k
                    
                    listvarmi = [] #liste des numeros de variants du grain m par rapport au grain i qui est le variant 1
                    listvarmj = [] #liste des numeros de variants du grain m par rapport au grain j  
                    listvarmk = [] #liste des numeros de variants du grain m par rapport au grain k 

                    for dmi in listOpmi:
                        lvarmi = listOpSource[dmi] # equivaut a FonctSource(dmi,1) mais en plus rapide
                        for varmi in lvarmi:
                            if varmi not in listvarmi: listvarmi.append(varmi)

                    listvarm = listvarmi

                    if listvarm: attr_varnum(m, listvarm)

    if listvoisvar:
        GrainGrowth2(doublet, listvoisvar, anggrowth)
    else:
        return 

#====================================================================================================

def GrainGrowth3(triplet, listgrexplor, anggrowth):

    global listIndGrdGrowth, TabGrains, listvoisvar
    
    # fait grossir le triplet de grains filles i=var1,j,k
    # listgrexplor ne contient que des grains qui n appartiennent pas deja a un grain parent
    
    #print "listgrexplor",listgrexplor
    #liste de grains deja testes

    listvoisvar = [] #liste des voisins des variants qui seront trouves
    # calcul de ceux qui sont lies aux grains i, j et k par des operateurs coherent

    #---------------------------------------------------------------------------------------------
    def attr_varnum(m, listvarm):
        global TabGrains, listvoisvar,listIndGrdGrowth
 
        for varm in listvarm:
            if varm not in TabGrains[m - 1].variant: TabGrains[m - 1].variant.append(varm)
        TabGrains[m - 1].confind = anggrowth
        listvoism = TabGrains[m - 1].listgrvois
        for voism in listvoism:
            if (len(TabGrains[voism - 1].variant) == 0) and (voism not in listvoisvar): listvoisvar.append(voism)

        if m not in listIndGrdGrowth: listIndGrdGrowth.append(m)
    #----------------------------------------------------

    try:
        i, j, k = triplet[0]
        dij, dik, djk = triplet[1]
    except: # si prend en fait un quadruplet a la place d un triplet
        i, j, k = triplet[0][:3]
        dij, dik, djk = triplet[1][:3]

    for m in listgrexplor:

        listdesmi = test(disorientationgrain(m, i), AngTolParentNucl) # si identite avec ang nucleation, plus facile
        listdesmj = test(disorientationgrain(m, j), AngTolParentNucl)
        listdesmk = test(disorientationgrain(m, k), AngTolParentNucl)

        if listdesmi == [0]: attr_varnum(m, TabGrains[i - 1].variant)
        elif listdesmj == [0]: attr_varnum(m, TabGrains[j - 1].variant)
        elif listdesmk == [0]: attr_varnum(m, TabGrains[k - 1].variant)
            
        else:
            listdesmi = test(disorientationgrain(m, i), anggrowth)
            listdesmj = test(disorientationgrain(m, j), anggrowth)
            listdesmk = test(disorientationgrain(m, k), anggrowth)

            if listdesmi and listdesmj and listdesmk:

                listOpmi1 = []
                for dmi in listdesmi:
                    for dmj in listdesmj:
                        if dij in TabCompoOp[dmi][dmj]: #coherence i,j,m
                            if dmi not in listOpmi1: listOpmi1.append(dmi)

                listOpmi2 = []
                for dmi in listdesmi:
                    for dmk in listdesmk:
                        if dik in TabCompoOp[dmi][dmk]: #coherence i,k,m
                            if dmi not in listOpmi2: listOpmi2.append(dmi)

                listOpmi = intersec(listOpmi1, listOpmi2)

                if listOpmi:    #repasse sur les grains i,j,k mais cette fois avec une precision sur les dmi potentiels
                                # va donc calculer les numeros des variants du grain m en fonction des numeros de variants des grains i,j,k
                    
                    listOpmj = []
                    for dmi in listOpmi:
                        for dmj in listdesmj:
                            if dij in TabCompoOp[dmi][dmj]: #coherence i,j,m
                                if dmj not in listOpmj: listOpmj.append(dmj)

                    listOpmk = []
                    for dmi in listOpmi:
                        for dmk in listdesmk:
                            if dik in TabCompoOp[dmi][dmk]: #coherence i,k,m
                                if dmk not in listOpmk: listOpmk.append(dmk)

                    listvarmi = [] #liste des numeros de variants du grain m par rapport au grain i qui est le variant 1
                    listvarmj = [] #liste des numeros de variants du grain m par rapport au grain j  
                    listvarmk = [] #liste des numeros de variants du grain m par rapport au grain k 

                    for dmi in listOpmi:
                        lvarmi = listOpSource[dmi] # equivaut a FonctSource(dmi,1) mais en plus rapide
                        for varmi in lvarmi:
                            if varmi not in listvarmi: listvarmi.append(varmi)

                    for dmj in listOpmj:
                        lvarmj = FonctSource(dmj, TabGrains[j - 1].variant[0])
                        for varmj in lvarmj:
                            if varmj not in listvarmj: listvarmj.append(varmj)

                    for dmk in listOpmk:
                        lvarmk = FonctSource(dmk, TabGrains[k - 1].variant[0])
                        for varmk in lvarmk:
                            if varmk not in listvarmk: listvarmk.append(varmk)

                    #print "growth", listvarmi,listvarmj,listvarmk

                    listvarm = intersec(intersec(listvarmi, listvarmj), listvarmk)

                    if listvarm: attr_varnum(m, listvarm)

    if listvoisvar:
        GrainGrowth3(triplet, listvoisvar, anggrowth)
    else:
        return 

 #====================================================================================================

def GrainGrowth4(quadruplet, listgrexplor, anggrowth):
    
    # fait grossir le triplet de grains filles i=var1,j,k
    # listgrexplor ne contient que des grains qui n appartiennent pas deja a un grain parent

    global listIndGrdGrowth, TabGrains, listvoisvar
    
    # fait grossir le triplet de grains filles i=var1,j,k
    # listgrexplor ne contient que des grains qui n appartiennent pas deja a un grain parent
    
    #print "listgrexplor",listgrexplor
    #liste de grains deja testes

    listvoisvar = [] #liste des voisins des variants qui seront trouves
    # calcul de ceux qui sont lies aux grains i, j et k par des operateurs coherent

    #---------------------------------------------------------------------------------------------
    def attr_varnum(m, listvarm):
        global TabGrains, listvoisvar,listIndGrdGrowth
 
        for varm in listvarm:
            if varm not in TabGrains[m - 1].variant: TabGrains[m - 1].variant.append(varm)
        TabGrains[m - 1].confind = anggrowth
        listvoism = TabGrains[m - 1].listgrvois
        for voism in listvoism:
            if (len(TabGrains[voism - 1].variant) == 0) and (voism not in listvoisvar): listvoisvar.append(voism)

        if m not in listIndGrdGrowth: listIndGrdGrowth.append(m)
    #----------------------------------------------------
        
    try:
        i, j, k, l = quadruplet[0]
        dij, dik, dil, djk, djl, dkl = quadruplet[1]
    except:
        try:
            i, j, k, l = quadruplet[0][:4]
            dij, dik, dil, djk, djl, dkl = quadruplet[1][:4]
        except:
            # le quadruplet est en fait un triplet qui vient dune reconstruction a 3 grains
            print "WARNINIG RECONSTRUCTION MODES MIXED with triplets and quadruplets"
            GrainGrowth3(quadruplet, listgrexplor, iparent2, anggrowth)
            return

    for m in listgrexplor:

        listdesmi = test(disorientationgrain(m, i), AngTolParentNucl) # si identite avec ang nucleation, plus facile
        listdesmj = test(disorientationgrain(m, j), AngTolParentNucl)
        listdesmk = test(disorientationgrain(m, k), AngTolParentNucl)
        listdesml = test(disorientationgrain(m, l), AngTolParentNucl)

        if listdesmi == [0]: attr_varnum(m, TabGrains[i - 1].variant)
        elif listdesmj == [0]: attr_varnum(m, TabGrains[j - 1].variant)
        elif listdesmk == [0]: attr_varnum(m, TabGrains[k - 1].variant)
        elif listdesml == [0]: attr_varnum(m, TabGrains[l - 1].variant)
        
        else:
            listdesmi = test(disorientationgrain(m, i), anggrowth)
            listdesmj = test(disorientationgrain(m, j), anggrowth)
            listdesmk = test(disorientationgrain(m, k), anggrowth)
            listdesml = test(disorientationgrain(m, l), anggrowth)

            if listdesmi and listdesmj and listdesmk and listdesml:

                listOpmi1 = []
                for dmi in listdesmi:
                    for dmj in listdesmj:
                        if dij in TabCompoOp[dmi][dmj]: #coherence i,j,m
                            if dmi not in listOpmi1: listOpmi1.append(dmi)

                listOpmi2 = []
                for dmi in listdesmi:
                    for dmk in listdesmk:
                        if dik in TabCompoOp[dmi][dmk]: #coherence i,k,m
                            if dmi not in listOpmi2: listOpmi2.append(dmi)

                listOpmi3 = []
                for dmi in listdesmi:
                    for dml in listdesml:
                        if dil in TabCompoOp[dmi][dml]: #coherence i,l,m
                            if dmi not in listOpmi3: listOpmi3.append(dmi)


                listOpmi = intersec(intersec(listOpmi1, listOpmi2), listOpmi3)

                if listOpmi:    #repasse sur les grains i,j,k,l mais cette fois avec une precision sur les dmi potentiels
                                # va donc calculer les numeros des variants du grain m en fonction des numeros de variants des grains i,j,k,l

                    listOpmj = []
                    for dmi in listOpmi:
                        for dmj in listdesmj:
                            if dij in TabCompoOp[dmi][dmj]: #coherence i,j,m
                                if dmj not in listOpmj: listOpmj.append(dmj)

                    listOpmk = []
                    for dmi in listOpmi:
                        for dmk in listdesmk:
                            if dik in TabCompoOp[dmi][dmk]: #coherence i,k,m
                                if dmk not in listOpmk: listOpmk.append(dmk)

                    listOpml = []
                    for dmi in listOpmi:
                        for dml in listdesml:
                            if dil in TabCompoOp[dmi][dml]: #coherence i,l,m
                                if dml not in listOpml: listOpml.append(dml)

                    #print "les listes op mi, mj, mk", listOpmi,listOpmj,listOpmk
                    listvarmi = [] #liste des numeros de variants du grain m par rapport au grain i qui est le variant 1
                    listvarmj = [] #liste des numeros de variants du grain m par rapport au grain j  
                    listvarmk = [] #liste des numeros de variants du grain m par rapport au grain k
                    listvarml = [] #liste des numeros de variants du grain m par rapport au grain l 

                    for dmi in listOpmi:
                        lvarmi = listOpSource[dmi] # equivaut a FonctSource(dmi,1) mais en plus rapide
                        for varmi in lvarmi:
                            if varmi not in listvarmi: listvarmi.append(varmi)

                    for dmj in listOpmj:
                        lvarmj = FonctSource(dmj, TabGrains[j - 1].variant[0])
                        for varmj in lvarmj:
                            if varmj not in listvarmj: listvarmj.append(varmj)

                    for dmk in listOpmk:
                        lvarmk = FonctSource(dmk, TabGrains[k - 1].variant[0])
                        for varmk in lvarmk:
                            if varmk not in listvarmk: listvarmk.append(varmk)

                    for dml in listOpml:
                        lvarml = FonctSource(dml, TabGrains[l - 1].variant[0])
                        for varml in lvarml:
                            if varml not in listvarml: listvarml.append(varml)

                    #print "les listes var mi, mj, mk, ml", listvarmi,listvarmj,listvarmk, listvarml

                    listvarm = intersec(intersec(intersec(listvarmi, listvarmj), listvarmk), listvarml)

                    if listvarm: attr_varnum(m, listvarm)

    if listvoisvar:
        GrainGrowth4(quadruplet, listvoisvar, anggrowth)
    else:
        return 

#====================================================================================================

def TRDGrowth(arrived, iparent2, anggrowth, order):
    
    # fait grossir listindgrains (= doublet apres nucleation) de grains filles 
    # listgrexplor ne contient que des grains qui n appartiennent pas deja a un grain parent
    # methode differente des parentgraingrowth, ici iteratif avec un ensemble de grains qui grossit

    global TabGrains, TabGrainsRec

    listvoisvar = [] #liste des voisins des variants qui seront trouves
    newarrived = []
    for i in arrived:
        # calcul de ceux qui sont lies aux grains i, j par des operateurs coherent
        for m in TabGrains[i - 1].listgrvois:
            if not TabGrains[m - 1].variant:
                t = testS3n(disorientationgrain(m, i), anggrowth, order) # a changer ensuite en testS3n
                if t != -1:
                    TabGrains[m - 1].variant = [0]
                    TabGrains[m - 1].grainp = iparent2 + 1 # A VERIFIER
                    TabGrains[m - 1].confind = anggrowth
                    TabGrainsRec[iparent2].append(m)
                    newarrived.append(m)

    if newarrived:
        TRDGrowth(newarrived, iparent2, anggrowth, order)
    else:
        return

#====================================================================================================

def ReconstTRD1 ():
    ReconstTRD(1)

def ReconstTRD2 ():
    ReconstTRD(2)

def ReconstTRD3 ():
    ReconstTRD(3)

def ReconstTRD4 ():
    ReconstTRD(4)
    
 #====================================================================================================

def ReconstTRD (order=1):
    global TabPix, TabGrains, TabParentGrains, listNuclei
    global Transition, Nbpixx, Nbpixy, XStep, Xmax, Ymax
    global OptShowA, OptShowB, ComptNucl, ComptGrowth, iparent, FlagRec
    global TabGrainsRec
    global OrderTwin, PercentRec, canvas

    # order = ordre de maclage limite pour arreter l identifcation ds operateurs

    OrderTwin = order

    EtatRun.set("\n"+"Nucleation with triplets. Running..."+"\n")
    LabelRun["fg"] = "red"
    frameFile.update()

    TimeStart = time.clock()

    if FlagRec == 1:
        FlagRec = 0
        #pour remttre a zero les compteurs
        TabGrainsRec = []
        listNuclei = []
        iparent = 0
        for i in range(len(TabPix)): TabPix[i][4] = 0 #4 = grainp
        for gr in TabGrains:
            gr.grainp = 0
            gr.variant = []

    if ComptNucl == 0:
        ComptNucl+=1
        istart = 0
        flagNucl = 1
        #print "----------------------NUCLEATION /FIRST GROWTH----------------------------"
        while flagNucl:
            TabGrainsRec.append([])
            #print "istart",istart
            res = TRDNucleation(istart, AngTolParentNucl,iparent)
            if res:
                i, j = res # i = grain trouve ayant nuclee, j  grain voisin en coherence sigma3 avec i
                listNuclei.append(res)
                TRDGrowth(res, iparent, AngTolParentGrowth, order)
                for newistart in range(i - 1, len(TabGrains)):
                    if not TabGrains[newistart].variant: break
                istart = newistart
                if istart == len(TabGrains) - 1: flagNucl = 0
                iparent = iparent + 1
            else:
                TabGrainsRec.pop(-1) #enleve le [] rajoute par defaut qui a avorte
                flagNucl = 0

    TRDFinish(AngTolParentNucl) #ajoute maintenant tous les grains restes celibataires

    UpDateProgress(0)

    #print "***********************************************************************"
    #print
    TabParentGrains = []
    for i in range(len(TabGrainsRec)):
        grparent = cloneGrain(TabGrains[TabGrainsRec[i][0] - 1]) #a faire: recalculer les angles Euler pour le grain parent! et les liste voisins
        grparent.grainp = i + 1
        grparent.graind = copy.deepcopy(TabGrainsRec[i])
        listvoisdaught = []
        #reattribut les voisins
        for j in range(len(TabGrainsRec[i])):
            lvois = TabGrains[TabGrainsRec[i][j] - 1].listgrvois
            for v in lvois:
                if v not in listvoisdaught: listvoisdaught.append(v)
        listvoispar = []
        for vd in listvoisdaught:
            for k in range(len(TabGrainsRec)):
                lgrp = TabGrainsRec[k]
                if vd in lgrp and k + 1 not in listvoispar and k != i: listvoispar.append(k + 1)
        grparent.listgrvois = listvoispar
        xp, yp = [0, 0]
        lgraind = []
        grparent.listpix = []
        #reattribut les pixels
        for j in range(len(TabGrainsRec[i])):
            lgraind.append(TabGrainsRec[i][j])
            listpixij = TabGrains[TabGrainsRec[i][j] - 1].listpix
            grparent.listpix.extend(listpixij)
            TabGrains[TabGrainsRec[i][j] - 1].grainp = i + 1
            for kk in listpixij: TabPix[kk][4] = i + 1 #4 = grainp
            xp = xp + TabGrains[TabGrainsRec[i][j] - 1].x
            yp = yp + TabGrains[TabGrainsRec[i][j] - 1].y
        grparent.graind = lgraind
        x, y = xp / len(TabGrainsRec[i]), yp / len(TabGrainsRec[i])
        grparent.x = x
        grparent.y = y
        j = int(x / XStep)
        if GRID != "Hexagon": i = int(y / XStep)
        else: i = int((y / (XStep*sqrt(3)/2)))
        nb = i * Nbpixx + j
        indgrdprim = TabPix[nb][3]
        if indgrdprim == -1:# sur des grains non convexe peut tomber sur des pixels non indexes => grain primaire = 1er grain
            flagind = 0
            for nbbis in Lvoisinspix4(nb):
                if TabPix[nbbis][3] != 0:
                    indgrdprim = TabPix[nbbis][3]
                    flagind = 1
                    break
            if not flagind: indgrdprim = TabGrainsRec[i][0]
        try:
            grprim = TabGrains[indgrdprim - 1]
            e1, e2, e3 = grprim.euler1, grprim.euler2, grprim.euler3
            grparent.euler1, grparent.euler2, grparent.euler3 = e1, e2, e3
            TabParentGrains.append(grparent)
        except:
            print "pb reconstrcution TRD", "lentabgrain=", len(TabGrains), "ind=", indgrdprim

    CalculateGrainBoundaries("grainp")

    #print " ***** fin procedure ReconstParentGrains3 *****"
    print "total duration=", (time.clock() - TimeStart) / 60, "min"
    print "Now import the ParentGrains file into the HKL software"

##    print "liste des grains parents"
##    for grp in TabParentGrains: print grp.toList()
##    print
    PercentRec = 0
    for grd in TabGrains:
        if grd.grainp:
            PercentRec += 1
    PercentRec = (100 * PercentRec) / len(TabGrains)

    ComptNucl+=1

    os.chdir (os.path.join(EBSDDirectory, "Results"))

##    print "****************"
##    print FileName
##    print os.getcwd()

    #SaveRecEnCours([TabGrainsRec,listNuclei, iparent])

    OptShowB = "grainp"
    OptShowA = "euler1"
    refreshButtons1()
    refreshButtons2()
    ButtonE["foreground"] = "white"
    ButtonE["background"] = "blue"
    ButtonN["foreground"] = "white"
    ButtonN["background"] = "blue"
    if not HideFlag.get(): showmap()

    EtatRun.set("\n"+"End " + str(PercentRec) + "% of grains reconstructed. If necessary, save the file."+"\n")
    LabelRun["fg"] = "blue"
    UpDateProgress(0)
    
 #===================================================================================================

def RecalcOrientVarNum():
    global TabOrient, TabGrains, TabParentGrains
   
    if len(TabParentGrains)!=len(TabGrainsRec): print "pb ici dans Recalc"
    AbsoluteVarNum() # 1st run to index the variants with the previous orientations of the parents, but with the new phase transition
    
    EtatRun.set("\n"+"Calculating the orientations of the parent grains..."+"\n")
    LabelRun["fg"] = "red"
    UpDateProgress(0)
    
    TabOrient=[]
    for i in range(len(TabGrainsRec)):
        listindgrd = TabGrainsRec[i]
        phi1,phi,phi2 = OneParentGrainOrientation(listindgrd,6)
        TabOrient.append([phi1,phi,phi2])
        grp = TabParentGrains[i]
        if phi1: grp.euler1, grp.euler2, grp.euler3 = phi1,phi,phi2
        
    AbsoluteVarNum() # 2d run to index the variants with the new orientations of the parents
 
    showmap()
     
#===================================================================================================

def AbsoluteVarNum():
    global TabGrains, TabParentGrains
    global NextStep, IndexQual

    EtatRun.set("\n"+"Indexing the variants..."+"\n")
    LabelRun["fg"] = "red"
    UpDateProgress(0)
        
    indqual = 0
    comptqual = 0
    stepg = max(len(TabParentGrains)/StProg,1)
    igrp = 0

    for grp in TabParentGrains:
        if not igrp%stepg: UpDateProgress((igrp * 100) / len(TabParentGrains))
        if grp.euler1 != 0: # euler1 = 0 pile veut dire quil na pas reussi a trouve lorientation du grain
            # calcul des orientations des variants theoriques
            angmoy = 0
            compt = 0
            PtoS = EulersToMat(grp.euler1, grp.euler2, grp.euler3,SourceFile) #S sample, P parent ortho frame
            StoP = dot(inv(PtoS), TSP) #P parent crystal
            listStoV = []
            for q in listQPos:
                StoV = dot(dot(StoP, q), PtoD)
                listStoV.append(StoV) #[matrice V1 to S,  matrice V2 to S ...]
            # identification des grains aux variants
            for indgrd in grp.graind:
                grd = TabGrains[indgrd - 1]
                grd.variant = []
                DtoS = EulersToMat(grd.euler1, grd.euler2, grd.euler3,SourceFile) #S sample, grd daughter grains
                DtoS = dot(invTSD, DtoS)
                lang = []
                for iv in range(len(listStoV)):
                    StoV = listStoV[iv]
                    dtov = dot(DtoS, StoV)
                    mdis = Disorientation(dtov, GroupeRotD)
                    if mdis == []: mdis = dtov
                    ang = testIdang(mdis)
                    lang.append(ang)
                angmin = min(lang) 
                angmoy += angmin
                compt += 1
                indvar = lang.index(angmin) + 1
                grd.variant = [indvar]
                if angmin<30: grd.confind = angmin
                else : grd.confind = 31 # pour garder une gamme de couleur dans l image
            if compt: angmoy = angmoy / compt
            else: angmoy = 30 # arbitraire
            grp.confind = angmoy
            #print "Parent Grain ", igrp+1, "Indexing angle dev.= ", angmoy, " Deg."
            indqual += angmoy
            comptqual += 1
        else:
            #print "Parent Grain non oriented"
            for indgrd in grp.graind:
                grd = TabGrains[indgrd - 1]
                grd.variant = []
                grp.confind = 30
        igrp += 1
    try:
        qual = cut(indqual / comptqual) + " Deg."
        IndexQual = indqual / comptqual
    except: qual = " Impossible. You need to Calculate the Orientations of the Parent Grains Before."
    NextStep += 1
    wd = "Quality Index= " + qual
    EtatRun.set("\n"+wd+"\n")
    LabelRun["fg"] = "blue"
    UpDateProgress(0)
    

#===================================================================================================

def CalcTriangleOR():
    global Transition, TransLoadMode
    global TabPixTriangle, OptShowA, frameMenuCV, ResTri

    TabPixTriangle = [[] for i in range(len(TabPix))]
    trans = copy.copy(Transition)

    EtatRun.set("\n"+"Loading Pitsch, NW, GT and KS ORs..."+"\n")
    LabelRun["fg"] = "red"
    UpDateProgress(0)

    ListPlanes = ListPlanesCubic
    TransLoadMode = 1
    
    # load the three phase transitions_________
    ListPlanes = ListPlanesCubic
    TransLoadMode = 1
    Transition = "Fe-Pitsch"
    LoadTransitionFiles(0)
    listQPos_Pitsch0 = listQPos
    PtoD_Pitsch = PtoD 
    print "Pitsch"
    
    ListPlanes = ListPlanesCubic
    TransLoadMode = 1
    Transition = "Fe-NW"
    LoadTransitionFiles(0)
    listQPos_NW0 = listQPos
    PtoD_NW = PtoD
    print "NW"

    ListPlanes = ListPlanesCubic
    TransLoadMode = 1
    Transition = "Fe-GT"
    LoadTransitionFiles(0)
    listQPos_GT0 = listQPos
    PtoD_GT = PtoD
    print "G"

    ListPlanes = ListPlanesCubic
    TransLoadMode = 1
    Transition = "Fe-KS"
    LoadTransitionFiles(0)
    listQPos_KS0 = listQPos
    PtoD_KS = PtoD
    print "KS"
    
    # come back to the initial OR ______________
    Transition = str(trans)
    print "Transition = ", Transition
    print "Come back on initial transition = ", Transition
    ListPlanes = ListPlanesCubic
    TransLoadMode = 1
    LoadTransitionFiles(0)
    
    #___________________________________________
    EtatRun.set("\n"+"Calculating the angle between exp. and theor. Pitsch - KS - NW and GT "+"\n")
    LabelRun["fg"] = "red"
    UpDateProgress(0)

    compt = 0
    ListTriAngMin = []
    stepg = max(len(TabGrains)/StProg,1)
    igrp = 0

    listQPos_Pitsch = []
    listQPos_KS = []
    listQPos_NW = []
    listQPos_GT = []
    for q in listQPos_Pitsch0: listQPos_Pitsch.append(dot(q, PtoD_Pitsch)) 
    for q in listQPos_KS0: listQPos_KS.append(dot(q, PtoD_KS)) 
    for q in listQPos_NW0: listQPos_NW.append(dot(q, PtoD_NW)) 
    for q in listQPos_GT0: listQPos_GT.append(dot(q, PtoD_GT))
    
    listQPos_Pitsch0 = []
    listQPos_KS0 = []
    listQPos_NW0 = []
    listQPos_GT0 = []        
    
    for grp in TabParentGrains:
        if not igrp%stepg: UpDateProgress((igrp * 100) / len(TabParentGrains))
        if grp.euler1 != 0: # euler1 = 0 pile veut dire quil na pas reussi a trouve lorientation du grain
            # calcul des orientations des variants theoriques
            PtoS = EulersToMat(grp.euler1, grp.euler2, grp.euler3,SourceFile) #S sample, P parent ortho frame
            StoP = dot(inv(PtoS), TSP) #P parent crystal
            listStoV_Pitsch = []
            listStoV_KS = []
            listStoV_NW = []
            listStoV_GT = []
            for q in listQPos_Pitsch:
                StoV = dot(StoP, q)
                listStoV_Pitsch.append(StoV) #[matrice S to V1,  matrice S to V2 ...]
            for q in listQPos_KS:
                StoV = dot(StoP, q)
                listStoV_KS.append(StoV) #[matrice S to V1,  matrice S to V2 ...]
            for q in listQPos_NW:
                StoV = dot(StoP, q)
                listStoV_NW.append(StoV) #[matrice S to V1,  matrice S to V2 ...]
            for q in listQPos_GT:
                StoV = dot(StoP, q)
                listStoV_GT.append(StoV) #[matrice S to V1,  matrice S to V2 ...]
            #------------------------------------------------------------------------
            for indgrd in grp.graind:
                if not compt%stepg: UpDateProgress((compt * 100) / len(TabGrains))
                grd = TabGrains[indgrd - 1]
                TriiMin = [] # liste des indices de variants approchants pour les trois ORs
                TriAngMin = []  # liste des angles mini pour les trois ORs
                # pour chaque transition P, KS et NW definit l indice du variant approchant imin
                # similaire  a absolute variant indexing
                # Triangle became Quadruple by introduction of GT (done latter)
                TriOp = [listStoV_Pitsch, listStoV_KS,listStoV_NW, listStoV_GT] # now 4 ORs(P, KS, NW and GT) and not only (P, KS, NW)
                for listStoVi in TriOp:
                    DtoS = EulersToMat(grd.euler1, grd.euler2, grd.euler3,SourceFile) #S sample, grd daughter grains
                    DtoS = dot(invTSD, DtoS)
                    angmin = 90
                    iMin = 0
                    for iv in range(len(listStoVi)): 
                        StoV = listStoVi[iv]
                        dtov = dot(DtoS, StoV)
                        mdis = Disorientation(dtov, GroupeRotD)
                        if mdis == []: mdis = dtov
                        ang = testIdang(mdis)
                        if ang<angmin:
                            angmin = ang
                            iMin = iv
                        if angmin<3: break
                    TriiMin.append(iMin)
                    TriAngMin.append(angmin)
                for indpix in grd.listpix: # calcule le RGB = P-KS-NW au niveau de chaque pixel
                    triangle=[]
                    #euler1, euler2, euler3 = pix 
                    DtoSpix = EulersToMat(TabPix[indpix][:3][0],TabPix[indpix][:3][1],TabPix[indpix][:3][2],SourceFile) #S sample, grd daughter grains
                    DtoSpix = dot(invTSD, DtoSpix)
                    for i in range(4):
                        iMin = TriiMin[i]
                        StoV = TriOp[i][iMin]
                        dtov = dot(DtoSpix, StoV)
                        mdis = Disorientation(dtov, GroupeRotD)
                        if mdis == []: mdis = dtov
                        ang = testIdang(mdis) 
                        triangle.append(ang)
                    TabPixTriangle[indpix]=triangle  
                ListTriAngMin.append(TriAngMin) # calcule les angles moyens au niveau des grains, pas au niveau des pixels
                compt+=1 # compteur de grains daughter
        igrp+=1

    if compt:
        angTriangle = transpose(array(ListTriAngMin))
        angP = average(angTriangle[0])
        angKS = average(angTriangle[1])
        angNW = average(angTriangle[2])
        angGT = average(angTriangle[3])
        qual = "Pitsch = " + cut(angP,3)+ " deg., KS = " + cut(angKS,3) + ", NW = " + cut(angNW,3)+" deg., GT = " + cut(angGT,3) + " deg."
    else: qual = " Impossible. You need to Calculate the Orientations of the Parent Grains Before."
    print "qual :", qual
    OptShowA == "Triangle"
    ResTri = "Indexing Quality: " + qual
    ShowMapTri(1)
    frameMenuCV.update()
    UpDateProgress(0)
    
#===================================================================================================

def calculateTabPixP():
    global TabPixP,OptShowB, OptShowA

    # calcule les orientations des pixels parents en supposant OR stricte

    t0 = time.clock()   
    EtatRun.set("\n"+"Calculating the orientations of parent pixels..."+"\n")
    LabelRun["fg"] = "red"
    UpDateProgress(0)

    GroupeRotPOrtho = []
    for g in GroupeRotP:
        GroupeRotPOrtho.append(dot(dot(TSP, g),invTSP))
    AngTolGrainP = 3* AngTolGrain
        
    #--------------------------------------
    def DisorientOrthoP(m):
        gm=[]
        for symk in GroupeRotPOrtho: 
            tr = trace(dot(symk,m))
            if tr < -1: tr = -1
            elif tr > 3: tr = 3
            arand = abs(arccos((tr - 1) / 2)) * 180. / pi
            if arand < AngTolGrainP:
                gm = symk
                break
        return gm
    #--------------------------------------
    def closeEulers(eul,listEulers):
        res=0
        i=1
        for euli in listEulers:
            if abs(eul[0]-euli[0]) + abs(eul[1]-euli[1]) + abs(eul[2]-euli[2]) < AngTolGrainP:
                res=i
                break
            i+=1
        return res   
    #--------------------------------------
    def findPv(mDtoS,mPtoS):
        #find the good variant of P from D to P transition
        res = []
        for mDtoPv in listMatDtoP: # Pv for variant of parents from D to P transition
            mStoPv = dot(inv(mDtoS), mDtoPv)
            mPtoPv = dot(mPtoS,mStoPv)
            mdis = Disorientation(mPtoPv, GroupeRotP)
            ang = testIdang(mdis)
            if abs(ang) < AngTolGrain:
                res = mDtoPv
                break
        return res            
    #--------------------------------------
        

    stepg = 1 #max(len(TabParentGrains)/2,1)
    igrp = 0
    for grp in TabParentGrains:
        #print "************************",igrp
        listAnalysed = []
        listDtoPv = [] # P = parent cryst, Do = daughter ortho
        mPtoS = dot(invTSP,EulersToMat(grp.euler1,grp.euler2,grp.euler3,SourceFile))
        if not igrp%stepg: UpDateProgress((igrp * 100) / len(TabParentGrains))
        if grp.euler1 != 0: # euler1 = 0 pile veut dire quil na pas reussi a trouve lorientation du grain
            for indgrd in grp.graind:
                grd = TabGrains[indgrd - 1]
                for indpix in grd.listpix:
                    if TabPix[indpix][5]!=300: # no retained parent at this place
                        # faire les calculs par zones de pixels
                        ie = closeEulers(TabPix[indpix][:3],listAnalysed)
                        mDtoS = dot(invTSD,EulersToMat(TabPix[indpix][:3][0],TabPix[indpix][:3][1],TabPix[indpix][:3][2],SourceFile))
                        if ie:
                            mDtoPv = listDtoPv[ie-1]
                            flag = 1
                        else:
                            mDtoPv = findPv(mDtoS,mPtoS)
                            listAnalysed.append(TabPix[indpix][:3])
                            listDtoPv.append(mDtoPv)
                        if len(mDtoPv):
                            mPvotoS = dot(TSP,dot(inv(mDtoPv),mDtoS)) # Pvo = Parent variant in ortho frame
                            TabPixP[indpix][0:3]= MatToEulers(mPvotoS,SourceFile,SourceFile) 
                        else: pass # equivalent a TabPixP[indpix]= [0,0,0]
        igrp+=1
        
    igrp = 0
    # decorelle les angles Euler equivalents
    for grp in TabParentGrains:
        if not igrp%stepg: UpDateProgress((igrp * 100) / len(TabParentGrains))
        if grp.euler1 != 0: # euler1 = 0 pile veut dire quil na pas reussi a trouve lorientation du grain
            i0 = grp.listpix[0]
            Ei = array([TabPixP[i0][0], TabPixP[i0][1], TabPixP[i0][2]])
            invMatEi = inv(EulersToMat(Ei[0],Ei[1],Ei[2],SourceFile))
            for indpix in grp.listpix[1:]:
                Eulers = TabPixP[indpix]
                MatEulers = EulersToMat(Eulers[0],Eulers[1],Eulers[2],SourceFile)
                m = dot(MatEulers, invMatEi)
                gm = DisorientOrthoP(m)
                if len(gm):
                    EEiv = MatToEulers(dot(gm,MatEulers,SourceFile))
                    d = add.reduce(map(abs, EEiv - Ei))
                    if d < AngTolGrainP:
                        TabPixP[indpix][0:3]= EEiv  #ici ici
        igrp+=1
 
    UpDateProgress(0)
    OptShowA = "euler1"
    OptShowB = "pixelsP"
    refreshButtons1()
    refreshButtons2()
    ButtonA["foreground"] = "white"
    ButtonA["background"] = "blue"
    ButtonNN["foreground"] = "white"
    ButtonNN["background"] = "blue"
    print "duration of parent pixel map = ", (time.clock()-t0)/60.
    showmap()

#===================================================================================================

def CheckStraight(lcoupleij):
    
    # define whetther the points are well aligned
    # lcoupleij = list of dots (i,j) with i creasing

    vect_S = []
    l = len(lcoupleij)
    pt1 = array(lcoupleij[0])
    pt2 = array(lcoupleij[l / 4])
    pt3 = array(lcoupleij[l / 2])
    pt4 = array(lcoupleij[3 * l / 4])
    pt5 = array(lcoupleij[-1])
    d31 = pt3 - pt1
    d35 = pt3 - pt5
    d34 = pt3 - pt4
    d32 = pt3 - pt2
    angrad = AngTolGBStraight * math.pi / 180

    if d31[0]: a31 = arctan(d31[1] / d31[0])
    else: a31 = math.pi / 2
    if abs(a31 + math.pi / 2) < angrad: a31 = math.pi / 2

    if d35[0]: a35 = arctan(d35[1] / d35[0])
    else: a35 = math.pi / 2
    if abs(a35 + math.pi / 2) < angrad: a35 = math.pi / 2

    if d34[0]: a34 = arctan(d34[1] / d34[0])
    else: a34 = math.pi / 2
    if abs(a34 + math.pi / 2) < angrad: a34 = math.pi / 2

    if d32[0]: a32 = arctan(d32[1] / d32[0])
    else: a32 = math.pi / 2
    if abs(a32 + math.pi / 2) < angrad: a32 = math.pi / 2

    amax = max(a31, a35, a34, a32)
    amin = min(a31, a35, a34, a32)
    if abs(amax - amin) < angrad:
        v = pt5 - pt1
        if GRID != "Hexagon": vect_S = array([v[0], v[1], 0]) # la longueur du vecteur = longueur du jdg
        else: vect_S = array([(sqrt(3) / 2) * v[0], v[1], 0])
    return vect_S

#===================================================================================================

def RecurCheckStraight(lcoupleij):
    pass
    # A FAIRE ICI

#===================================================================================================

def DrawGBSigma3n3():
    DrawGBSigma3n(3)

def DrawGBSigma3n4():
    DrawGBSigma3n(3)

#def DrawGBSigma3n4():
#    DrawGBSigma3n(4)

def DrawGBSigma3n5():
    DrawGBSigma3n(5)

def DrawGBSigma3n6():
    DrawGBSigma3n(6)

#===================================================================================================

def DrawGBSigma3n(order=3):
    
    global canvas, ListOpGBLines, OrderTwin, ListOpGBPercL, ListOpGBPercU
    global s3count, s3ncount, countL, countU

    OrderTwin = order
    ListOpGBLines = [[] for n in range(len(listOpDisor) + 1)]
    ListOpGBPercL = [0 for n in range(len(listOpDisor) + 1)]
    ListOpGBPercU = [0 for n in range(len(listOpDisor) + 1)]

    di = 1
    dj = 1
    s3count = 0 # nb pixels GB en sigma3
    s3ncount = 0 # nb pixels GB en sigma3n
    countL = 0 # nb pixels GB total
    countU = 0 # nb GB (unitaire) total

    EtatRun.set("\n"+"Calculating Sigma3n boundaries..."+"\n")
    LabelRun["fg"] = "red"
    frameFile.update()

    #------------------
    def drawGBInList(listindGrd, flagUpD, listexclugrd=[]):
        global canvas, ListOpGBLines, ListOpGBPercL, ListOpGBPercU, s3count, s3ncount, countL, countU

        for indgrd in listindGrd:
            
            grd = TabGrains[indgrd - 1]
            if flagUpD:
                listgrdvois = grd.listgrvois
            else:
                listgrdvois = copy.deepcopy(grd.listgrvois)
                for indgrdexc in listexclugrd:
                    while indgrdexc in listgrdvois: listgrdvois.remove(indgrdexc)

            for indgrdvois in listgrdvois:
                nopvoisgrd = testS3n(disorientationgrain(indgrd, indgrdvois), AngTolGrain, order) #-1 rien, 0 identity, 1 sigma 3, 2 sigma9 etc.
                nop = nopvoisgrd + 1 #0 rien, 1 identity, 2 sigma 3, 3 sigma9 etc.
                grdvois = TabGrains[indgrdvois - 1]
                listpixfrcom = intersecpix(grd.listpixfr, grdvois.listpixfr)
                ListOpGBPercU[nop] += 1
                countU += 1

                if nop == 0:ncol = 0
                else:
                    ncol = 0
                    for noplim in listTwinNum[:order + 1]: # ne garde que lordre de maclage
                        if noplim > nopvoisgrd: break
                        ncol += 1
                lij = []

                for pix in listpixfrcom:
                    i = pix / Nbpixx
                    j = pix - Nbpixx * i
                    lij.append([i, j])

                    if not TwinFlag_pol.get(): col = COLORS_AMB[ncol]
                    else:
                        if n < 5: col = COLORS_AMB[ncol]
                        else:
                            t = listTwinNom[nop][0]
                            if '+' in t: col = COLORS_POL[5 + 2 * (ncol - 5)]
                            elif '-' in t: col = COLORS_POL[6 + 2 * (ncol - 5)]
                            else: col = COLORS_AMB[ncol]

                    if GRID != "Hexagon":
                        gbd = canvas.create_rectangle(int(j * Scale), int(i * Scale), int((j + dj) * Scale), int((i + di) * Scale), fill=col, outline="")
                    else:
                        gbd = canvas.create_rectangle(int(j * Scale), int(i * Scale * sqrt(3) / 2),
                                    int((j + dj) * Scale), int((i + di) * Scale * sqrt(3) / 2), fill=col, outline="")
                    canvas.itemconfig(gbd, tags="GBD")
                    if nop == 2: s3count += 1 # un pixel GB en sigma3
                    if nop >= 2: s3ncount += 1 # un pixel GB en sigma3n

                l = len(listpixfrcom)
                ListOpGBPercL[nop] += l
                countL += l
                if l > 5:
                    lij.sort()
                    vect_S = CheckStraight(lij) # S = sample
                    if vect_S != []:
                        m_CtoS = EulersToMat(grd.euler1, grd.euler2, grd.euler3,SourceFile) #transf matrix crystal ref cubique to sample
                        #m_CtoS = dot(inv(TSD),m_CtoS) #transf matrix crystal to sample, ici = identity
                        vect_C = dot(m_CtoS, vect_S)
                        ListOpGBLines[nop].append([l * XStep, vect_C]) # straight, l*XStep = longueur
                    else: ListOpGBLines[nop].append([l * XStep]) #  not straight, but we keeponly the length
                #elif l>0: ListOpGBLines[n].append([0]) # too short
    #------------------

    if len(TabParentGrains) == 0: drawGBInList(range(1, len(TabGrains) + 1), 1) # peut etre fait avant la reconstruction des TRD

    else: # permet  de ne prendre en compte que les joints interieurs aux TRDs
        stepg = max(len(TabParentGrains)/StProg,1)
        indgrp = 1
        for grparent in TabParentGrains:
            listexclugrd = []
            if not indgrp%stepg: UpDateProgress((indgrp * 100) / len(TabGrains))
            # cherche les jdg filles qui bordent le jdg parent
            for indgrd in grparent.graind:
                grd = TabGrains[indgrd - 1]
                for indgrdvois in grd.listgrvois:
                    if indgrdvois not in grparent.graind:
                        if indgrdvois not in listexclugrd: listexclugrd.append(indgrdvois)
                        ListOpGBPercU[0] += 1
                        countU += 1
                        grdvois = TabGrains[indgrdvois - 1]
                        frpixcom = intersecpix(grd.listpixfr, grdvois.listpixfr)
                        ListOpGBLines[0].append([len(frpixcom) * XStep])
                        l = len(frpixcom)
                        ListOpGBPercL[0] += l
                        countL += l

            drawGBInList(grparent.graind, 0, listexclugrd) # trace l interieur d un grain parent en exluant les grainsd de listexclugrd
            if not stepg: UpDateProgress((indgrp * 100) / len(TabParentGrains))
            indgrp += 1

    ParentGB.set(1)
    showGBP()

    ListOpGBPercL = map(lambda x: float(100 * x) / countL, ListOpGBPercL)
    ListOpGBPercU = map(lambda x: float(100 * x) / countU, ListOpGBPercU)

    print ListOpGBPercL
    print
    print ListOpGBPercU
    print "--"

    EtatRun.set("\n"+"Sigma3 = " + str((100 * s3count) / countL) + " % , Sigma3n with n>0 = " +
                str((100 * s3ncount) / countL) + " %"+"\n")
    LabelRun["fg"] = "blue"
    DaughterGB.set(1)
    UpDateProgress(0)
    frameFile.update()
    try : winsound.PlaySound("!", winsound.SND_ALIAS)
    except: pass

#===================================================================================================

def DrawSigma3Aust():
    
    global canvas

    nx = Nbpixx
    ny = Nbpixy
    di = 2
    dj = 2

    EtatRun.set("\n"+"Drawing Sigma3 boundaries in Reconstructed Austenite..."+"\n")
    LabelRun["fg"] = "red"
    frameFile.update()

    stepg = len(TabParentGrains)/StProg

    for indgrp in range(1,len(TabParentGrains)+1):
        grp = TabParentGrains[indgrp-1]
        #if not indgrp%stepg: UpDateProgress((indgrp * 100) / len(TabParentGrains))
        for indgrvois in grp.listgrvois:
            frsigma =[]
            if testS3(disorientationgrain(indgrp, indgrvois, OptShowB), AngTolGrain):
                grpvois=TabParentGrains[indgrvois-1]
                frsigma = intersecpix(grp.listpixfr,grpvois.listpixfr)
            for pix in frsigma :
                    i = pix / nx
                    j = pix - nx * i
                    col = "red"
                    if GRID != "Hexagon": gbps = canvas.create_rectangle(int(j * Scale), int(i * Scale), int((j + dj) * Scale), int((i + di) * Scale), fill=col, outline="")
                    else:
                        gbps = canvas.create_rectangle(int(j * Scale), int(i * Scale * sqrt(3) / 2),
                                                      int((j + dj) * Scale), int((i + di) * Scale * sqrt(3) / 2), fill=col, outline="")
                    canvas.itemconfig(gbps, tags="GBPS")

    if TabRetainedGrains:
        for indgrp in range(1,len(TabRetainedGrains)+1):
            grp = TabRetainedGrains[indgrp-1]
            #if not indgrp%stepg: UpDateProgress((indgrp * 100) / len(TabParentGrains))
            for indgrvois in grp.listgrvois:
                frsigma =[]
                if indgrp!= -1:
                    if testS3(disorientationgrain(-indgrp, -indgrvois, OptShowB), AngTolGrain): # les indices sont negatifs pour retained parent
                        grpvois=TabRetainedGrains[abs(indgrvois)-1]
                        frsigma = intersecpix(grp.listpixfr,grpvois.listpixfr)
                    for pix in frsigma :
                            i = pix / nx
                            j = pix - nx * i
                            col = "red"
                            if GRID != "Hexagon": gbps = canvas.create_rectangle(int(j * Scale), int(i * Scale), int((j + dj) * Scale), int((i + di) * Scale), fill=col, outline="")
                            else:
                                gbps = canvas.create_rectangle(int(j * Scale), int(i * Scale * sqrt(3) / 2),
                                                              int((j + dj) * Scale), int((i + di) * Scale * sqrt(3) / 2), fill=col, outline="")
                            canvas.itemconfig(gbps, tags="GBPS")
                else: pass

    EtatRun.set("\n"+"Reconstructed Parent Grains, with Sigma3 GBs drawn in red."+"\n")
    LabelRun["fg"] = "blue"
    UpDateProgress(0)
    frameFile.update()
    try: winsound.PlaySound("!", winsound.SND_ALIAS)
    except: pass

#===================================================================================================

def DrawGBOp():
    
    global canvas, ListOpGBLines, ListOpGBPercL, ListOpGBPercU

    ListOpGBLines = [[] for n in range(len(listOpDisor) + 1)]
    ListOpGBPercL = [0 for n in range(len(listOpDisor) + 1)]
    ListOpGBPercU = [0 for n in range(len(listOpDisor) + 1)]

    nx = Nbpixx
    ny = Nbpixy
    di = 1
    dj = 1
    countL = 0
    countU = 0

    EtatRun.set("\n"+"Calculating Grain boundaries..."+"\n")
    LabelRun["fg"] = "red"
    frameFile.update()

    stepg = len(TabGrains)/StProg
    indgrd = 1
    for grd in TabGrains:
        if not indgrd%stepg: UpDateProgress((indgrd * 100) / len(TabGrains))
        ldes = []
        lgrvois = []
        lopgrvois = []
        listLines = [[[] for nop in range(len(listOpDisor) + 1)] for iv in grd.listgrvois]
        #chaque grain voisin definit une frontiere, nop + 1 car 0 pour Random
        for indgrvois in grd.listgrvois:
            lgrvois.append(indgrvois)
            nopvoisgrd = test(disorientationgrain(indgrd, indgrvois), AngTolGrain) #liste des operateurs possibles
            #-1 rien, 0 identity, nop>0 other operators
            if nopvoisgrd: nopvoisgrd = nopvoisgrd[0]
            else: nopvoisgrd = -1
            lopgrvois.append(nopvoisgrd)
            ListOpGBPercU[nopvoisgrd + 1] += 1
            countU += 1

        for pix in grd.listpixfr:
            pixgrd = TabPix[pix]
            lpixv = Lvoisinspix4(pix)
            flag = 0
            for pixv in lpixv:
                ivois = TabPix[pixv][3]
                if ivois in lgrvois:
                    iv = lgrvois.index(ivois)
                    nop = lopgrvois[iv] #-1 rien, 0 identity, nop>0 other operators
                    n = nop + 1
                    i = pix / nx
                    j = pix - nx * i
                    listLines[iv][n].append([i, j])
                    flag = 1
                    break

            if flag:
                col = COLORS_OP[n]
                if GRID != "Hexagon": gbd = canvas.create_rectangle(int(j * Scale), int(i * Scale), int((j + dj) * Scale), int((i + di) * Scale), fill=col, outline="")
                else:
                    gbd = canvas.create_rectangle(int(j * Scale), int(i * Scale * sqrt(3) / 2),
                                int((j + dj) * Scale), int((i + di) * Scale * sqrt(3) / 2), fill=col, outline="")
                canvas.itemconfig(gbd, tags="GBD")
                countL += 1
                ListOpGBPercL[n] += 1

        for grlij in listLines:
                for n in range(len(grlij)):
                    lcoupleij = grlij[n]
                    l = len(lcoupleij)
                    if l > 5:
                        lcoupleij.sort()
                        vect_S = CheckStraight(lcoupleij) # S = sample
                        if vect_S != []:
                            m_CtoS = EulersToMat(grd.euler1, grd.euler2, grd.euler3,SourceFile) #transf matrix crystal ref cubique to sample
                            m_CtoS = dot(inv(TSD), m_CtoS) #transf matrix crystal to sample
                            vect_C = dot(m_CtoS, vect_S)
                            ListOpGBLines[n].append([l * XStep, vect_C]) # straight
                        else: ListOpGBLines[n].append([l * XStep]) # we keep the length
                    #elif l>0: ListOpGBLines[n].append([0]) # too short

        indgrd += 1

    if ParentGB.get() and TabParentGrains: showGBP()

    ListOpGBPercL = map(lambda x: float(100 * x) / countL, ListOpGBPercL)
    ListOpGBPercU = map(lambda x: float(100 * x) / countU, ListOpGBPercU)

    EtatRun.set("\n"+"Grain boundaries = Operators, done."+"\n")
    LabelRun["fg"] = "blue"
    DaughterGB.set(1)
    UpDateProgress(0)
    frameFile.update()
    try: winsound.PlaySound("!", winsound.SND_ALIAS)
    except: pass

#====================================================================================================

def ShowGBPlanes():
    
    global ListGBPlanes, tabGB, angtol
    # tabGB = repartition des GB selon les plans cristallins operateur par operateur
    # colGBOp = col pour column, repartion des GB selon Op (GB unity and linear) [[u,l] pour random, [] pour op 0 etc.]

    ListGBPlanes = copy.deepcopy(ListPlanes)
    count = 0
    tabGB = []

    if ListOpGBLines == []:
        EtatRun.set("\n"+"Before, you need to Draw the special grain boundaries!"+"\n")
        LabelRun["fg"] = "red"

    angtol = AngTolGBStraight / 2

    fpar = Toplevel(root)

    lt = Label(fpar, foreground="black")
    lt["text"] = "List of Planes to be compared to straight GBs :"
    lt.grid(row=10, column=1, columnspan=2, sticky=E + W)

    listboxop = Listbox(fpar, width=10, height=10, selectmode=EXTENDED)
    for i in range(len(ListGBPlanes)):
        listboxop.insert(END, "{" + str(ListGBPlanes[i])[1:-1] + "}")
    listboxop.grid(row=11, column=1, ipadx=2, sticky=E + W)
    listboxop.select_set(0, END)

    fadd = Frame(fpar)

    fadd.grid(row=11, column=2, ipadx=2, sticky=E + W)
    fhkl = Frame(fadd)
    fhkl.grid(row=1, column=1, sticky=E + W)
    lt = Label(fhkl, text="{", foreground="black").grid(row=1, column=1, sticky=E + W)
    lt = Label(fhkl, text="}", foreground="black").grid(row=1, column=5, sticky=E + W)
    entry_paramGB = [None for i in range(3)]
    for j in range(3):
        entry_paramGB[j] = Entry(fhkl, width=3, foreground="blue")
        entry_paramGB[j].insert(0, 0)
        entry_paramGB[j].grid(row=1, column=j + 2, sticky=E + W)

    def AddToList():
        global ListGBPlanes
        h = string.atoi(entry_paramGB[0].get())
        k = string.atoi(entry_paramGB[1].get())
        l = string.atoi(entry_paramGB[2].get())
        ListGBPlanes.append([h, k, l])
        listboxop.insert(END, "  {" + str([h, k, l])[1:-1] + "}")
        listboxop.select_set(0, END)

    b = Button(fadd, text="Add to List", command=AddToList)
    b.grid(row=2, column=1, sticky=E + W)
    b["foreground"] = "black"
    b["background"] = "grey"

    lt = Label(fadd, text=" ", height=1, foreground="black").grid(row=3, column=1, sticky=E + W)

    lt = Label(fadd, foreground="black")
    lt["text"] = "Tolerance for \n plane identif. (Deg.)"
    lt.grid(row=4, column=1, sticky=E + W)

    entry_AngTol = Entry(fadd, width=5, foreground="blue")
    entry_AngTol.insert(0, angtol)
    entry_AngTol.grid(row=5, column=1, sticky=E + W)

    #-------------------------------
    def calctab():
        global ListGBPlanes, tabGB

        tabGB = []
        listlEq = GenerateEquiv(ListGBPlanes, GroupeRotD)
        listlEqGNorm = []
        nbeqmax = 0
        for lgplane in listlEq:
            leqg = []
            for g in lgplane: leqg.append(normalize(g))
            listlEqGNorm.append(leqg)
            if len(lgplane) > nbeqmax: nbeqmax = len(lgplane) #nb de plans equivalents au maximum dans la liste

        for n in range(len(ListOpGBLines)):
            if Transition == "Sigma3" and n > listTwinNum[OrderTwin + 1]: break
            listgbu = [0 for i in range(len(ListGBPlanes) + 2)]
            listgbustat = [0 for i in range(len(ListGBPlanes) + 2)]
            listlenmax = [[0] for i in range(len(ListGBPlanes) + 2)]
            listlenstat = [0 for i in range(len(ListGBPlanes) + 2)]
            listang = [0 for i in range(len(ListGBPlanes) + 2)]
            listvectC = ListOpGBLines[n]

            for longAndVect in listvectC:
                flag = 1
                if len(longAndVect) == 1: # not straight or too short, but we have kept the length of the gb
                    lC = longAndVect[0]
                    listgbu[0] += 1
                    listgbustat[0] += 1. / nbeqmax
                    listlenmax[0].append(lC)
                    listlenstat[0] += lC / nbeqmax
                    flag = 0
                else:
                    lC, vectC = longAndVect
                    for ng in range(len(listlEqGNorm)):
                        listGNorm = listlEqGNorm[ng]
                        for gplane in listGNorm: # list des vecteurs g equivalents et normalises
                            a = arccos(dot(normalize(vectC), gplane)) * 180. / math.pi
                            if abs(a + 90) < AngTolGBStraight or abs(a - 90) < angtol :
                                flag = 0
                                listgbu[ng + 1] += 1
                                listgbustat[ng + 1] += 1. / len(listGNorm) #calcul probabiliste, on divise par le nb de plans equivalents
                                listlenmax[ng + 1].append(lC)
                                listlenstat[ng + 1] += lC / len(listGNorm)
                                listang[ng + 1] += min(abs(a + 90), abs(a - 90))
                                break
                    if flag:
                        listgbu[-1] += 1
                        listgbustat[-1] += 1. / nbeqmax
                        listlenmax[-1].append(lC)
                        listlenstat[-1] += lC / nbeqmax


            nb = len(listvectC)
            totstat = add.reduce(listgbustat)
            if nb > 0 and totstat > 0:
                lGBu = map(lambda x:float(100 * x) / nb, listgbu)
                lGBuStat = map(lambda x:float((100 * x) / totstat), listgbustat)
                totlenStat = add.reduce(listlenstat)
                lGBlenStat = map(lambda x:(100 * x) / totlenStat, listlenstat)
                llenmax = map(lambda l:max(l), listlenmax)
                lang = []
                for  k in range(len(listang)):
                    if listgbu[k]: lang.append(listang[k] / listgbu[k])
                    else: lang.append(0)
            else:
                lGBuStat, lGBlenStat, llenmax, lang = [], [], [], []

            tabGB.append([lGBuStat, lGBlenStat, llenmax, lang])

    #-------------------------------
    def showgbtab(opt=0):
        try: ftab.destroy()
        except: pass
        try: bgraph.destroy()
        except: pass

        def showhisto():
            if opt == 0:
                listnbop = ListOpGBPercU
                titre = "GB Statistics. GB = Unity"
            else:
                listnbop = ListOpGBPercL
                titre = "GB Statistics. GB = Linear"
            UpdateStatistic([], listnbop, -10, titre)

        bgraph = Button(fpar, text="Histogram", command=showhisto)
        bgraph.grid(row=29, column=1, sticky=E + W)
        bgraph["foreground"] = "red"
        bgraph["background"] = "grey"

        ftab = Frame(fpar)
        ftab.grid(row=30, column=1, columnspan=2, sticky=E + W)

        if opt == 0: p = ListOpGBPercU[0]
        else: p = ListOpGBPercL[0]
        t = cut(p, 3)
        lt = Label(ftab, text=t + " % ", foreground="black", background="white").grid(row=1, column=0, sticky=E + W)
        lt = Label(ftab, text="Random" + "  ", foreground="black", background="grey").grid(row=1, column=1, sticky=E + W)
        if Transition != "Sigma3":
            for i in range(len(listOpDisor)):
                if opt == 0: p = ListOpGBPercU[i + 1]
                else: p = ListOpGBPercL[i + 1]
                t = cut(p, 3)
                lt = Label(ftab, text=t + " % ", foreground="black", background="white").grid(row=i + 2, column=0, sticky=E + W)
                lt = Label(ftab, text="Op. " + str(i) + "  ", foreground="black", background="grey").grid(row=i + 2, column=1, sticky=E + W)
        else:
            for i in range(len(listTwinNom)):
                if i >= listTwinNum[OrderTwin + 1]:break
                if opt == 0: p = ListOpGBPercU[i + 1]
                else: p = ListOpGBPercL[i + 1]
                nom = listTwinNom[i][0]
                t = cut(p, 3)
                lt = Label(ftab, text=t + " % ", foreground="black", background="white").grid(row=i + 2, column=0, sticky=E + W)
                lt = Label(ftab, text="Sigma3_" + nom + "  ", foreground="black", background="grey").grid(row=i + 2, column=1, sticky=E + W)

        lt = Label(ftab, text="NA", foreground="black", background="grey").grid(row=0, column=2, sticky=E + W)
        for j in range(len(ListGBPlanes)):
            lt = Label(ftab, text="{" + str(ListGBPlanes[j])[1:-1] + "}", foreground="black", background="grey").grid(row=0, column=3 + j, sticky=E + W)
        lt = Label(ftab, text="Other", foreground="black", background="grey").grid(row=0, column=len(ListGBPlanes) + 3, sticky=E + W)
        n = 0
        for sl in tabGB:
            lustat, llenstat, llen, lang = sl
            j = 0
            if opt == 0:      ll = [cut(lustat[k], 3) for k in range(len(lustat))]
            elif opt == 1:    ll = [cut(llenstat[k], 3) for k in range(len(llenstat))]
            elif opt == 2:    ll = [cut(llen[k], 3) for k in range(len(llen))]
            elif opt == 3:    ll = [cut(lang[k]) for k in range(len(lang))]

            for j in range(len(ll)):
                t = str(ll[j])
                lt = Label(ftab, text=t, foreground="black").grid(row=1 + n, column=2 + j, sticky=E + W)
                j += 1
            n += 1
    #-------------------------------

    def calcandshowgbtab():
        calctab()
        showgbtab(0)

    def showGBUp():
        global angtol
        for b in [b1, b2, b3, b4]:  b["foreground"] = "black"
        b1["foreground"] = "blue"
        angtoln = string.atof(entry_AngTol.get())
        if angtoln != angtol:
            angtol = angtoln
            calctab()
        showgbtab(0)
        linfo["text"] = "Values = % of grain boundary numbers"

    def showGBLp():
        global angtol
        for b in [b1, b2, b3, b4]:  b["foreground"] = "black"
        b2["foreground"] = "blue"
        angtoln = string.atof(entry_AngTol.get())
        if angtoln != angtol:
            angtol = angtoln
            calctab()
        showgbtab(1)
        linfo["text"] = "Values = % of grain boundary lengths"

    def showGBL():
        global angtol
        for b in [b1, b2, b3, b4]:  b["foreground"] = "black"
        b3["foreground"] = "blue"
        angtoln = string.atof(entry_AngTol.get())
        if angtoln != angtol:
            angtol = angtoln
            calctab()
        showgbtab(2)
        linfo["text"] = "Values = maximum grain boundary length in " + Units

    def showAng():
        global angtol
        for b in [b1, b2, b3, b4]:  b["foreground"] = "black"
        b4["foreground"] = "blue"
        angtoln = string.atof(entry_AngTol.get())
        if angtoln != angtol:
            angtol = angtoln
            calctab()
        showgbtab(3)
        linfo["text"] = "Standard deviation in Deg. of the \n angle(gb line,cryst. plane) from 90 Deg."

    b = Button(fpar, text="ShowTable", command=calcandshowgbtab)
    b.grid(row=20, column=1, columnspan=2, sticky=E + W)

    fb = Frame (fpar)
    fb.grid(row=22, column=1, columnspan=4, sticky=E + W)

    b1 = Button(fb, text="% GB Unity", command=showGBUp)
    b1.grid(row=0, column=1, sticky=E + W)
    b1["foreground"] = "blue"
    b1["background"] = "grey"

    b2 = Button(fb, text="% GB Length", command=showGBLp)
    b2.grid(row=0, column=2, sticky=E + W)
    b2["foreground"] = "black"
    b2["background"] = "grey"

    b3 = Button(fb, text="Max. Length", command=showGBL)
    b3.grid(row=0, column=3, sticky=E + W)
    b3["foreground"] = "black"
    b3["background"] = "grey"

    b4 = Button(fb, text="Ang. Dev.", command=showAng)
    b4.grid(row=0, column=4, sticky=E + W)
    b4["foreground"] = "black"
    b4["background"] = "grey"

    lt = Label(fpar, height=2, foreground="black")
    lt["text"] = "____________________________________________"
    lt.grid(row=23, column=1, columnspan=2, sticky=E + W)

    calctab()
    showgbtab(0)

    lt = Label(fpar, height=2, foreground="black")
    lt["text"] = "____________________________________________"
    lt.grid(row=49, column=1, columnspan=2, sticky=E + W)

    linfo = Label(fpar)
    linfo.grid(row=50, column=1, columnspan=2, sticky=E + W)
    linfo["text"] = "Values = % of grain boundary numbers"

#===================================================================================================

def TransfTabParentTabRec(): # si on veut reprendre la reconstruction a partir d anciennes
    
    global TabGrainsRec, TabOrient, FlagRec

    TabGrainsRec = []
    TabOrient = []
    for grp in TabParentGrains:
        TabGrainsRec.append(grp.graind)
        TabOrient.append([grp.euler1,grp.euler2,grp.euler3])
    
    if TabParentGrains: FlagRec = 0

#===================================================================================================

def LoadResults():
    
    global TabGrains, TabParentGrains, TabPix, TabGrainsRec, TabOrient, FlagRec
    global Transition, Nbpixx, Nbpixy, XStep, Xmax, Ymax
    global euler1min, euler2min, euler3min, bcmin, deuler1, deuler2, deuler3, dbc
    global SourceFile

    EtatRun.set("\n"+"Loading..."+"\n")
    LabelRun["fg"] = "red"
    frameFile.update()

    os.chdir (EBSDDirectory + "/Results")

    fname = str(FileName + ".arpge")
    f = open(fname, 'r')
    info = pickle.load(f)
    Transition, Nbpixx, Nbpixy, XStep, Xmax, Ymax, euler1min, euler2min, euler3min, deuler1, deuler2, deuler3, bcmin, dbc = info[0]
    InfoAdd = info[1]
    if "TSL" in InfoAdd: SourceFile = "ARPGE_TSL"
    if "HKL" in InfoAdd: SourceFile = "ARPGE_HKL"
    f.close()

    
    TabPix= []
    fname = str(FileName + "_TabPix.txt")
    try: # ouverture du fichier compresse
        f = gzip.open(fname+'.gz', 'rb')
        contents = f.read()
        f.close()
        lines = string.split(contents, '\n')
        for l in lines:
            try:
                ls = string.split(l,"\t")
                pix = map(float,ls[:3]) # e1,e2,e3,indgrd, indgrp, bc : avec en float les 3 premiers et les autres en int
                pix.extend(map(int,ls[3:6]))
                TabPix.append(pix)
            except: pass # le fichier se termine par un ''
            
    except:  # Ancienne methode sans compression  
        print ".gz file could not be found"
        fname = str(FileName + "_TabPix.txt")
        f = open(fname, 'r')
        TabPix = pickle.load(f)
        f.close()

    fname = str(FileName + "_TabGrains.txt")
    f = open(fname, 'r')
    TabGrains = pickle.load(f)
    f.close()

    try: # tente ouverture Grains Parents                
        fname = str(FileName + "_TabParentGrains.txt")
        f = open(fname, 'r')
        TabParentGrains = pickle.load(f)
        f.close() # pour l instant il manque les pixels
        # atribution des pixels aux grains parents
        for grp in TabParentGrains:
            listpixd = []
            for indgrd in grp.graind:
                grd = TabGrains[indgrd-1]
                listpixd.extend(grd.listpix)
            grp.listpix = listpixd
        TransfTabParentTabRec()
    except :
        print "Parent Grain File not found."

    EtatRun.set("\n"+"File Loaded."+"\n")
    LabelRun["fg"] = "blue"

#===================================================================================================
    
def SaveResults():
    SaveResults0(False,True)
 
#===================================================================================================
    
def SaveResults0(asksave = True,askoverw = True):
    
    global FileName

    os.chdir (os.path.join(EBSDDirectory, "Results"))

    ffname = str(FileName+ ".arpge")
    save = True
    saveParent = True
    overw = False

    if asksave:
        TimeStart = time.clock()
        save = tkMessageBox.askyesno('Warning', 'Do you want to save the results?')

    if save:
        if askoverw:
            if ffname in os.listdir(os.path.join(EBSDDirectory, "Results")):
                overw = tkMessageBox.askyesno('Warning', 'Overwrite the existing file' + FileName + ' ?')

        if not overw:
            if FileName[-2]=="v" and FileName[-3]=="_":
                try: ind = int(FileName[-1])
                except: ind =0
                if ind>=9: ind =0
            else:
                ind =0
                FileName = FileName + "_v"+str(0)
            ind+=1
            FileName = FileName[:-1] + str(ind)
            ffname = str(FileName + ".arpge")

        EtatFile.set(FileName)
        EtatSize.set(str(Nbpixx) + " x " + str(Nbpixy) + " pixels")

        EtatRun.set("\n"+"Saving..."+"\n")
        LabelRun["fg"] = "red"
        frameFile.update()

        ffname = str(FileName + ".arpge")
        ff = open(ffname, 'w')
        Info = [[Transition, Nbpixx, Nbpixy, XStep, YStep, Xmax, Ymax, euler1min, euler2min, euler3min, deuler1, deuler2, deuler3, bcmin, dbc],
                "file created by ARPGE software written by C. Cayron. Initial file = "+SourceFile, time.asctime()]
        pickle.dump(Info, ff)
        ff.close()

        fname = str(FileName + "_TabGrains.txt")
        f = open(fname, 'w')
        pickle.dump(TabGrains, f)
        f.close()

        if saveParent:
            TabParentGrainsRed = []
            for grp in TabParentGrains:
                grpred = cloneGrain(grp)
                grpred.listpix=[]
                TabParentGrainsRed.append(grpred)
                
            fname = str(FileName + "_TabParentGrains.txt")
            f = open(fname, 'w')
            pickle.dump(TabParentGrainsRed, f) # on a exclu la liste de pixel pour reduire la taille du fichier
            f.close()
        
        try: # sauve en version compresse
            X,Y = 0,0
            XMax = Nbpixx * XStep
            YMax = Nbpixy * YStep
            content=''
            for nb in range(len(TabPix)):
                E1 = TabPix[nb][0] 
                E2 = TabPix[nb][1] 
                E3 = TabPix[nb][2] 
                indgrd,indgrp = TabPix[nb][3],TabPix[nb][4]
                BC = TabPix[nb][5]
                s = '%3.4f \t %3.4f \t %3.4f \t %3d \t %4d \t %4d \n' % (E1, E2, E3, indgrd, indgrp, BC)
                X = X + XStep
                if X >= XMax:
                    X = 0
                    Y = Y + YStep
                content = content + s
                os.chdir (EBSDDirectory + "/Results")
            f = gzip.open(str(FileName + '_TabPix.txt.gz'), 'wb')
            f.write(content)
            f.close()
        except:
            print "pb for saving compressed version"
            fname = str(FileName + "_TabPix.txt")
            f = open(fname, 'w')
            pickle.dump(TabPix, f)
            f.close()

        EtatRun.set("\n"+"File saved."+"\n")
        LabelRun["fg"] = "blue"
        os.startfile(EBSDDirectory + "/Results")

#===================================================================================================    
#===================================================================================================

#====================================================================================================

def RemoveSpines():
    global ShapeFact      
    ShapeFact = 0.7
    RemoveSpinesIterate(0)

#====================================================================================================

def RemoveSpinesRecurs():
    global ShapeFact
    ShapeFact = 0.7
    RemoveSpinesIterate(1)
    
#====================================================================================================

def RemoveSpinesIterate(rec):
    # remove the daughter grains in a parent grain at its frontaries in function of their shape
    # they are removed if their common frontaries > certain percent 
    
    global TabGrains, TabParentGrains, TabPix, NextStep, ShapeFact

    EtatRun.set("\n"+"Suppressing Daughter Spine Grains..."+"\n")
    LabelRun["fg"] = "red"
    frameFile.update()
    
    #-----------------------
    def suppressGrd(indgrd,listpixfrcom=[],front_grd_ext=[]):
        # retire un graind de son grain parent et le met a zero
        global TabGrains, TabParentGrains, TabPix
        grd = TabGrains[indgrd-1]
        indgrp = grd.grainp
        grp = TabParentGrains[indgrp-1]
        grd.grainp = 0
        grd.variant=[]
        grd.confind = 30 # valeur arbitrairement haute (en deg)
        grp.graind.remove(indgrd)
        setpixgrd = set(grd.listpix)
        setpixgrp = set(grp.listpix)
        grp.listpix = list(setpixgrp-setpixgrd)
##        for pix in grd.listpix:
##            try: grp.listpix.remove(pix) # ancienne methode sans passer par SET,  cetait trop long
##            except: pass #print "error in suppressGrd"
        for pix in grd.listpix: TabPix[pix][4] = 0
        if not listpixfrcom:
            listpixfrcom, front_grd_ext= intersec2(grd.listpixfr,grp.listpixfr)
        for pixfrc in listpixfrcom:
            try: grp.listpixfr.remove(pixfrc)
            except: pass #print "error 2 in suppressGrd"
        grp.listpixfr.extend(front_grd_ext)
    #-----------------------
            
    #print "RemoveSpinesIterate"
    compt = 0
    L = len(TabParentGrains)
    stepg = max(L/StProg,1)
    flag2 = 0
    for indgrp in range(1,len(TabParentGrains)+1):
        grp = TabParentGrains[indgrp-1]
        if not compt%stepg: UpDateProgress((compt * 100) / L)
        front_grp = grp.listpixfr
        #listindgrd = copy.deepcopy(grp.graind)
        # ne prend que les grains de la frontiere
        listindgrd = []
        for indpix in front_grp:
            indgrd = TabPix[indpix][3]
            if indgrd not in listindgrd: listindgrd.append(indgrd)
        lindgrdeff=[]
        for indgrd in listindgrd:
            grd = TabGrains[indgrd - 1]
            front_grd = grd.listpixfr
            listpixfrcom, front_grd_ext= intersec2(front_grd,front_grp)
            indgrp = grd.grainp
            if indgrp:
                if len(listpixfrcom)> ShapeFact*len(front_grd):
                    #--------- efface le graind -----------------------------------
                    suppressGrd(indgrd,listpixfrcom,front_grd_ext)
                    lindgrdeff.append(indgrd)
                    flag2 = 1
                    # retire ensuite les graind qui se retrouvent isoles
##                    
##                    for indgrdvois in grd.listgrvois :
##                        grdvois = TabGrains[indgrdvois-1]
##                        if grdvois.grainp==indgrp:
##                            c3 = 0
##                            for indgrdvoisvois in grdvois.listgrvois:
##                                if TabGrains[indgrdvoisvois-1].grainp==indgrp:
##                                    c3+=1 # compteur de grains de meme couleur
##                            if c3<NbMinGrd-1:# condition arbitraire  
##                                if indgrdvois not in lindgrdeff: suppressGrd(indgrdvois)                    
        compt+=1
        
    if flag2 and rec:
        ShapeFact+=0.15
        UpDateProgress(0)
        RemoveSpinesIterate(rec)
    else:
        # efface les grains parents vide
        listcr=[]
        for indgrp in range(len(TabParentGrains)):
            grp = TabParentGrains[indgrp]
            if len(grp.graind)==0:
                listcr.append(indgrp+1)
                TabParentGrains[indgrp]=0
                #print "remove ", indgrp+1
        while 0 in TabParentGrains: TabParentGrains.remove(0)
        #----------
        if listcr:
            for grd in TabGrains:
                grd.grainp = renum(grd.grainp,listcr)
            for grp in TabParentGrains:
                grp.grainp = renum(grp.grainp,listcr)
                grp.listgrvois = [renum(x,listcr) for x in grp.listgrvois]
                while 0 in grp.listgrvois: grp.listgrvois.remove(0)
            for pix in TabPix: pix[4]= renum(pix[4],listcr)
        #----------------
        TransfTabParentTabRec()
        CalculateGrainBoundaries("grainp")
        if not HideFlag.get(): showmap()
        UpDateProgress(0)
        NextStep += 1
        EtatRun.set("\n"+"Daughter Spine Grains Removed"+"\n")
        LabelRun["fg"] = "blue"
        frameFile.update()

#===================================================================================================
        
def RemWhiteHoles():
    global ShapeFact
    ShapeFact = 0.8
    RemWhiteHolesIterate(0)
    
#===================================================================================================
        
def RemWhiteHolesRecurs():
    global ShapeFact
    ShapeFact = 0.8
    RemWhiteHolesIterate(1)
    
#===================================================================================================
        
def RemWhiteHolesIterate(rec):
    
    global TabGrains, TabParentGrains, TabPix, NextStep, ShapeFact

    EtatRun.set("\n"+"Suppressing White Holes..."+"\n")
    LabelRun["fg"] = "red"

    listgrdholes = []
    #---------------------
    for indgrd in range(len(TabGrains)):
        grd = TabGrains[indgrd-1]
        if not grd.grainp: listgrdholes.append(indgrd)
    #----------------------
    flag2 = 0
    compt = 0
    L = len(listgrdholes)
    stepg = max(L/StProg,1)
    for indgrd in listgrdholes:
        if not compt%stepg: UpDateProgress((compt * 100) / L)
        listindgrpvois=[]
        listindgrdvois = TabGrains[indgrd - 1].listgrvois
        flag = 0   
        for indgrdvois in listindgrdvois :
            indgrp = TabGrains[indgrdvois - 1].grainp
            if indgrp: flag = 1
            if indgrp not in listindgrpvois: listindgrpvois.append(indgrp)
        if flag:
            indgrpvoisR = 0
            grd = TabGrains[indgrd - 1]
            front_grd = grd.listpixfr
            lmax = 0
            lpixcf=[]
            for indgrpvois in listindgrpvois:
                lpixc = []
                longtot=0
                if indgrpvois>0:
                    lpixc = intersecpix(TabParentGrains[indgrpvois-1].listpixfr,front_grd)
                    longtot = len(lpixc)
                else:
                    listindgrdvois = TabGrains[indgrd - 1].listgrvois
                    for indgrd2 in listindgrdvois:
                        if TabGrains[indgrd2-1].grainp==0:
                            l = intersecpix(TabGrains[indgrd2-1].listpixfr,front_grd)
                            longtot+=len(l)
                            lpixc.extend(l)
                if longtot> lmax:
                    indgrpvoisR  = indgrpvois
                    lmax = longtot
                    lpixcf = lpixc
            if indgrpvoisR and lmax > ShapeFact*len(front_grd):
                flag2 = 1
                grpvois = TabParentGrains[indgrpvoisR - 1]
                listpixij = TabGrains[indgrd - 1].listpix
                grpvois.listpix.extend(listpixij)
                grpvois.graind.append(indgrd)
                TabGrains[indgrd - 1].grainp = indgrpvoisR
                TabGrains[indgrd - 1].confind = 30
                TabGrains[indgrd - 1].variant=[]
                for nindgrdvois in TabGrains[indgrd - 1].listgrvois: #03/08/10
                    nindgrpvois = TabGrains[nindgrdvois - 1].grainp
                    if nindgrpvois  and nindgrpvois != indgrpvoisR and nindgrpvois not in grpvois.listgrvois: grpvois.listgrvois.append(nindgrpvois)
                for indpix in listpixij: TabPix[indpix][4] = indgrpvoisR
                for indpix in lpixcf:
                    try: grpvois.listpixfr.remove(indpix)
                    except:pass
                for indpix in front_grd:
                    if indpix not in lpixcf: grpvois.listpixfr.append(indpix)
        compt+=1

    if rec and flag2:
        #ShapeFact+=0.1
        RemWhiteHolesIterate(rec)
    else:
        TransfTabParentTabRec()
        CalculateGrainBoundaries("grainp")
        if not HideFlag.get(): showmap()
        UpDateProgress(0)
        EtatRun.set("\n"+"Daughter White Holes Removed"+"\n")
        LabelRun["fg"] = "blue"
        NextStep += 1
        return

#===================================================================================================

def ResetListGrd(listindgrd):

    global TabGrains

    for indgrd in listindgrd:
        grd = TabGrains[indgrd-1]
        grd.grainp = 0
        grd.variant = []
        grd.confind = 30

#===================================================================================================

def ResetRec():

    global ComptNucl, ComptGrowth, FlagRec
    global TabGrainsRec, listNuclei, PercentRec, PercentOrient
    global TabGrains, TabParentGrains, TabPix, TabPixP, NextStep, iparent
    global AngTolGrain, AngTolParentGrowth, AngTolParentNucl
    global canvas, ComptNucl, ComptGrowth 
    global TabOrient, listIndGrdGrowth, TabPixTriangle

    canvas.delete(ALL)
    ComptNucl, ComptGrowth = 0, 0
    TabPixTriangle = []

    if Transition != "Sigma3":
        AngTolGrain = 5. #degree
        AngTolParentGrowth = 6. #degree
        AngTolParentNucl = 3. #degree
        EtatAngTolGrain.set(AngTolGrain)
        EtatAngTolParentGrowth.set(AngTolParentGrowth)
        EtatAngTolParentNucl.set(AngTolParentNucl)
    TabGrainsRec = []
    listNuclei = []
    TabOrient= []
    listIndGrdGrowth = []
    iparent = 0
    FlagRec = 1
    PercentRec = 0
    PercentOrient = 0
    RecMethod.set(0) #  methode triplet
    ComptNucl, ComptGrowth = 0, 0
    for grd in TabGrains:
        grd.grainp = 0
        grd.variant = []
        grd.confind = 30
    TabParentGrains = []
    for pix in TabPix:
        pix[4] = 0
    for pix in TabPixP:
        pix[4] = 0
    NextStep = 5
    if not HideFlag.get(): showmap()

#===================================================================================================

def checkInternalSymP(listmat1, listmat2): 
    #listmat = liste de matrices D_REF=>P possibles
    #verifie dans la deux listes listmat celles qui sont communes par symmetries internes de la phase parent
    # et renvoie la nouvelle liste d intersection

    if len(listmat1)==1:
        return listmat1
    else:
        angtol = AngTolGrain
        listmatres = []
        for mat1 in listmat1:
            for mat2 in listmat2:
                mat12 = dot(inv(mat1), mat2)
                flag = 0
                for gp in GroupeRotP:
                    try:
                        mk = dot(mat12, gp)
                        angle = abs(arccos((trace(mk) - 1) / 2)) * DEGREE
                        if abs(angle) < angtol:
                           listmatres.append(mat1)
                           flag = 1
                           break
                    except : pass
                if flag: break
        return  listmatres

#===================================================================================================

def buildOReq(mat1, listlistmat2): 
    #matref = matrices D_REF=>P trouvee par checkInternalSymP
    # listlist2 = list pour chaque graind d une liste de matrices possibles pour OR grp
    # et renvoie la nouvelle liste dans listmat2 des matrices les plus proches de matref et fait le modulo la de symetrie

    angtol = AngTolGrain
    listmatres = [mat1]
    for listmat2 in listlistmat2[1:]:
        flag = 0
        for mat2 in listmat2:
            mat12 = dot(inv(mat1), mat2)
            for gp in GroupeRotP:
                try:
                    mk = dot(mat12, gp)
                    angle = abs(arccos((trace(mk) - 1) / 2)) * DEGREE
                    if abs(angle) < angtol:
                       listmatres.append(dot(mat2,gp))
                       flag = 1
                       break
                except : pass
            if flag: break
             
    return  listmatres
#===================================================================================================

def OneParentGrainOrientation(listgrd, nbvarlim=4,trimode=0):
    # calculate the orientation of the parent grain from a list of daughter grains
    # trimode = 0: tri par indice de confiance confinf, =1 : tri par taille de grains
    
    res = 0,0,0 #phi1, phi, phi2 of parent grain
    
    nbGrFound = 0   
    quadrupletGrd = []
    quadrupletVar = []
    listindgrorder = []
    if not trimode: # tri sur les indices de confiance
        for indgrd in listgrd:
            ci = TabGrains[indgrd - 1].confind
            listindgrorder.append([ci, indgrd])
        listindgrorder.sort() #tri selon les indices de confidence 0=good 30 =bad
    else: # tri par taille de grain
        for indgrd in listgrd:
            ci = len(TabGrains[indgrd - 1].listpix)
            listindgrorder.append([ci, indgrd])
        listindgrorder.sort(reverse=True) #tri selon les tailles du plus grand au plus petit
        
    if not trimode: 
        for cind in listindgrorder:
            indgrdi = cind[1]
            lvar = TabGrains[indgrdi - 1].variant
            if len(lvar) == 1: #cherche que les grains ou les numeros de variants ont ete donnes sans ambiguite
                v = lvar[0]
                if v not in quadrupletVar: #cherche 4 numeros de variants tous distincts
                    quadrupletVar.append(v)
                    quadrupletGrd.append(indgrdi)
            if len(quadrupletVar) >= nbvarlim: break # en cours: puis ne pas limiter a quatre?
    if len(quadrupletVar)<4: # alors on reboucle en etant moins regardant sur lambiguite
        for cind in listindgrorder:
            indgrdi = cind[1]
            lvar = TabGrains[indgrdi - 1].variant
            for v in lvar:
                if v not in quadrupletVar: #cherche 4 numeros de variants tous distincts
                    quadrupletVar.append(v)
                    quadrupletGrd.append(indgrdi)
                    break
            if len(quadrupletVar) >= nbvarlim: break # en cours: puis ne pas limiter a quatre?
    QuadrupletListORgrP = []
    #print quadrupletGrd,quadrupletVar
    for indgrd in quadrupletGrd:
        grd = TabGrains[indgrd - 1]
        ORgrD = inv(EulersToMat(grd.euler1, grd.euler2, grd.euler3,SourceFile))
        ORgrD = dot(ORgrD, TSD) # = transformation matrix [REF=>grd]
        listORgrP = []
        for matDtoP in listMatDtoP:
            listORgrP.append(dot(ORgrD, matDtoP))
        QuadrupletListORgrP.append(listORgrP)
    if len(QuadrupletListORgrP)>3: listORgrP1 = reduce(checkInternalSymP, QuadrupletListORgrP)
    elif len(QuadrupletListORgrP)>2: listORgrP1 = reduce(checkInternalSymP, QuadrupletListORgrP)
    elif len(QuadrupletListORgrP)>=1: listORgrP1 = reduce(checkInternalSymP, QuadrupletListORgrP)
    else: listORgrP1=[]
    # modifie en janv 2020 pour permettre reconst NiTi
    if Transition== "NiTi": cond = (len(listORgrP1)>=1)
    else: cond = (len(listORgrP1)==1)
    if cond:
        nbGrFound+=1
        listORgrP = buildOReq(listORgrP1[0],QuadrupletListORgrP)
        c =0
        phi1m, phim, phi2m = 0,0,0
        for ORgrP in listORgrP:
            ORgrP = dot(ORgrP, inv(TSP)) 
            mPtoREF = inv(ORgrP)
            phi1, phi, phi2 = MatToEulers(mPtoREF,SourceFile)
            phi1m, phim, phi2m = phi1m + phi1, phim + phi, phi2m + phi2
            c+=1
        res = phi1m/c,phim/c,phi2m/c # ose moyenner les angles d'Euler
    else: pass
      
    return res

#===================================================================================================
def cherchechTwin(nop, mat, angtol):
    #cherche le chemin ch = mat
    
    res = []
    listmat = []
    invmat = inv(mat)
    for s in GroupeRotD:
        listmat.append(dot(s,invmat))# je calcule toutes les symmetries internes, => mult a droite, ici a gauche car inv pour comparaison
    listch = listTwinNom[nop][1]
    flag = 0
    for ch in listch:
        OrVar = CodeTwinToMatrix(ch)
        for m in listmat:
            residu = dot(m, OrVar)
            tr = trace(residu)
            ad = abs(arccos((tr - 1) / 2)) * DEGREE
            if ad < angtol:
                res = ch
                flag = 1
                break
        if flag:break
    if not flag: print "could not found a twin"
    return res

#===================================================================================================

def TRDTrees():
    
    global TabPix, TabGrains, TabParentGrains, canvas
    global TRDTreeDone
    
    EtatRun.set("\n"+"Calculation of the trees. Running...."+"\n")
    LabelRun["fg"] = "red"
    invTSD = inv(TSD)
    #----------------------------------
    def ProgressTree(indgrdprim, listgrd, iparent):
        global TabGrains
        listgrexplor = []
        for indgrd in listgrd:
            grd = TabGrains[indgrd - 1]
            nop, ch = grd.variant
            for indgrvois in grd.listgrvois:
                if TabGrains[indgrvois - 1].grainp == iparent and TabGrains[indgrvois - 1].variant == [0]:
                    if indgrvois not in listgrexplor: listgrexplor.append(indgrvois)
            newlistgrd = []
            for indgrvois in listgrexplor: # liste des grains a explorer
                nopvoisgrd = testS3n(disorientationgrain(indgrd, indgrvois), AngTolParentGrowth, OrderTwin)
                if nopvoisgrd != -1:
                    if nop < len(listTwinCompo) and nopvoisgrd < len(listTwinCompo):
                        try: listnomopvois = listTwinCompo[nop][nopvoisgrd]
                        except:
                            print "error in TRDTrees (tabcompo twin)", nop, nopvoisgrd
                            listnomopvois = []
                        m_Pgrdvois = disorientationgrain(indgrdprim, indgrvois)
                        nopvois = testS3nInList(m_Pgrdvois, listnomopvois, AngTolParentGrowth)
                        if nopvois != -1: # il est possible que ca renvoit -1 si la matrice correspond a un sigma3n avec n>6 (on pourrait aller jusqua 8 en modifiant loadtransitionsigma)
                            chvois = cherchechTwin(nopvois, m_Pgrdvois, AngTolParentGrowth) #cherche le chemin chvois pour que m_Pgrdvois = nopvois
                            TabGrains[indgrvois - 1].variant = [nopvois, chvois]
                            newlistgrd.append(indgrvois)

        if newlistgrd:  ProgressTree(indgrdprim, newlistgrd, iparent) # on reprend avec listgrexplor comme nouvelle liste de base
        else: return
    #----------------------------------                 
    indparent = 1
    stepg = max(len(TabParentGrains)/StProg,1)
    for grparent in TabParentGrains:
        j = int(grparent.x / XStep)
        if GRID != "Hexagon": i = int(grparent.y / YStep)
        else: i = int(grparent.y / (XStep*sqrt(3)/2))
        nb = i * Nbpixx + j
        indgrdprim = TabPix[nb][3]
        grprim = TabGrains[indgrdprim - 1]
        grprim.variant = [0, ''] # numero de l operateur et chaine correspondante           
        ProgressTree(indgrdprim, [indgrdprim], indparent)
        if not indparent%stepg: UpDateProgress((indparent * 100) / len(TabParentGrains))
        indparent += 1

    OptShowB = "grainp"
    OptShowA = "euler1"
    refreshButtons1()
    refreshButtons2()
    ButtonE["foreground"] = "white"
    ButtonE["background"] = "blue"
    ButtonN["foreground"] = "white"
    ButtonN["background"] = "blue"
    if not HideFlag.get(): showmap()

    EtatRun.set("\n"+"TRD trees done. "+"\n")
    LabelRun["fg"] = "blue"
    UpDateProgress(0)
    TRDTreeDone = 1

    return

#===================================================================================================

def ModePoleParentGrain():

    global OptShowB
    global PolFigEulers
    
    if OptShowB != "polefig" and OptShowB != "polefig2": OptShowB = "polefig"
    else : OptShowB = "grainp"

    EtatRun.set("\n"+"Double-click on the parent grain (Rechoose Indiv.Parent Grain to Stop)"+"\n")
    LabelRun["fg"] = "blue"

    if OptShowB == "polefig" or OptShowB == "polefig2" :
        ButtonPF["foreground"] = "white"
        ButtonPF["background"] = "blue"
    else:
        ButtonPF["foreground"] = "black"
        ButtonPF["background"] = "grey"

    from arpge.Graphics.CC_PlotPoleFigure     import PolFigEulers

#===================================================================================================

def ModePoleParentGrain2():

    global OptShowB
    global PolFigEulers
    
    if OptShowB != "polefig2" and OptShowB != "polefig": OptShowB = "polefig2"
    else : OptShowB = "grainp"

    EtatRun.set("\n"+"Double-click on the parent grain (Rechoose Indiv.Parent Grain to Stop)"+"\n")
    LabelRun["fg"] = "blue"

    if OptShowB == "polefig2" or OptShowB == "polefig":
        ButtonPF["foreground"] = "white"
        ButtonPF["background"] = "blue"
    else:
        ButtonPF["foreground"] = "black"
        ButtonPF["background"] = "grey"

    from arpge.Graphics.CC_PlotPoleFigure     import PolFigEulers

#===================================================================================================

def ModeStatisticParentGrain():

    global OptShowB, CVStatistic, CVStatistic2
    
    if OptShowB != "statistic":
        OptShowB = "statistic"
        EtatRun.set("\n"+"Double-click on the parent grain (Rechoose Indiv.Parent Grain to Stop)"+"\n")
        LabelRun["fg"] = "blue"
    else :
        EtatRun.set("\n"+"Idle"+"\n")
        LabelRun["fg"] = "blue"
        OptShowB = "grainp"
        try: CVStatistic.delete(ALL)
        except: pass
        try: CVStatistic2.delete(ALL)
        except:pass

#===================================================================================================

def ModeFractal():

    global OptShowB, CVStatistic, CVStatistic2

    if not TRDTreeDone :
        TRDTrees()
        EtatRun.set("\n"+"Calculating TRD Trees..."+"\n")
        LabelRun["fg"] = "red"

    if OptShowB != "fractal":
        OptShowB = "fractal"
        EtatRun.set("\n"+"Double-click on the parent grain (Rechoose Indiv.Parent Grain to Stop)"+"\n")
        LabelRun["fg"] = "blue"
    else :
        EtatRun.set("\n"+"Idle"+"\n")
        LabelRun["fg"] = "blue"
        OptShowB = "grainp"
        CVStatistic.delete(ALL)
        CVStatistic2.delete(ALL)
        
#===================================================================================================
        
def CalcAnisotropy(gr):
    # calcule l anisotrope d un grain
    # 1 calcule la distance la plus grande GX = a avec X appart a frontiere
    # 2 calule b par l aire s = pi.a.b
    # 4 r = a / b

    def possimpl(pos):
        i = pos / Nbpixx
        j = pos - i * Nbpixx
        if GRID != "Hexagon": x,y = j,i 
        else: x,y = j, i *(sqrt(3)/2)
        return array([x,y])
        
    def dist(v):
        return sqrt(v[0]**2+v[1]**2)

    G = array([gr.x/XStep,gr.y/YStep]) # suppose YStep = XStep
    dmax = 0
    for pixfr in gr.listpixfr:
        X = possimpl(pixfr)
        d = dist(G-X)
        if d>dmax: dmax = d
    a = dmax
    s = len(gr.listpix)
    b = s/(math.pi*a)
    try: r = a/b
    except: r = 0
           
    return r      
              
#===================================================================================================

def HistoGrainSizesD():
    
    global NextStep
    
    HistoGrainSizes('graind')
    NextStep += 1

#===================================================================================================

def HistoGrainSizesP():
    
    HistoGrainSizes('grainp')

#===================================================================================================

def HistoGrainSizes(opt='graind', scalebar="um", step=1, dmin=0, dmax=0, CVSize=200,areafreq=0,anisotrop=0,amin=0,amax=0):
    # dmin et dmax taille des grains
    # amin et amax anisotropie des grains
    
    global fHisto, fHisto2, Buttonf, CVHisto, Areafreq, Anisotrop

    try: Buttonf.destroy()
    except:pass

    try: fHisto.destroy()
    except:pass

    try: fHisto2.destroy()
    except:pass

    try: CVHisto.destroy()
    except:pass

    def exportHisto():
        try:
            d = getResultsPath()+ '\Results'
            os.chdir(d)
        except:
            d = getResultsPath()
            os.chdir(d)
        filehist = open("histogram_GrainSize.txt","w")
        filehist.write("step "+str(step)+"\n")
        for f in listf:
            filehist.write(str(f*100)+"\n")
        filehist.close()
        tkMessageBox.showinfo("info", "the text file of the histogram is saved in the directory: " + d)
        os.startfile(d)

    def saveHisto():
        try:
            v = CVHisto.find_withtag("value1")
            CVHisto.delete(v)
            v = CVHisto.find_withtag("value2")
            CVHisto.delete(v)
        except: pass
        try:
            d = getResultsPath()+ '\Results'
            os.chdir(d)
        except:
            d = getResultsPath()
            os.chdir(d)

        WC,HC = CVHisto.winfo_width(), CVHisto.winfo_height()
        x1,y1 = CVHisto.winfo_rootx(), CVHisto.winfo_rooty()
        ImageGrab.grab((x1,y1,x1+WC,y1+HC)).save(filename + '.bmp')
        tkMessageBox.showinfo("info", "the image of the histogram is saved in the directory: " + d)
        os.startfile(d)

    def OpenCVHisto(event):
        HistoGrainSizes(opt, scalebar, step, dmin, dmax, CVSize,areafreq,anisotrop)

    def CloseCVHisto(event):
        global Buttonf, fHisto
        fHisto.destroy()
        fHisto2.destroy()
        try: CVHisto.destroy()
        except:pass
        Buttonf = Button(frameHisto, text="Grain Size \n Statistics")
        Buttonf["background"] = "grey"
        Buttonf.bind("<Button>", OpenCVHisto)
        Buttonf.grid (row=0, column=0, ipadx=1, sticky='W')
        frameHisto.update()

    def attr_histvar(event):
        step = string.atof(EntryStepHisto.get())
        dmin = string.atof(EntryMin.get())
        dmax = string.atof(EntryMax.get())
        CVSize = string.atoi(EntryCVSize.get())
        CVHisto.destroy()
        HistoGrainSizes(opt, scalebar, step, dmin, dmax, CVSize,Areafreq, Anisotrop)

    def attr_histvarA(event):
        step = string.atof(EntryStepHisto.get())
        amin = string.atof(EntryMin.get())
        amax = string.atof(EntryMax.get())
        CVSize = string.atoi(EntryCVSize.get())
        CVHisto.destroy()
        HistoGrainSizes(opt, scalebar, step, dmin, dmax, CVSize,Areafreq, Anisotrop,amin,amax)

    def AreaFreq(event):
        global Areafreq
        Areafreq=1
        attr_histvar(0)

    def GrainFreq(event):
        global Areafreq
        Areafreq=0
        attr_histvar(0)

    def AnisotropFreq(event):
        global Anisotrop
        if not Anisotrop:
            EntryMin.delete(0,END)
            EntryMax.delete(0,END)
            EntryMin.insert(0, cut(amin))
            EntryMax.insert(0, cut(amax))  
        Anisotrop=1
        attr_histvar(0)

    def MeanSizeFreq(event):
        global Anisotrop
        if Anisotrop:
            EntryMin.delete(0,END)
            EntryMax.delete(0,END)
            EntryMin.insert(0, cut(dmin))
            EntryMax.insert(0, cut(dmax))  
        Anisotrop=0
        attr_histvar(0)
    
    if opt == 'graind': TabG = TabGrains
    elif opt == 'grainp': TabG = TabParentGrains

    Areafreq = areafreq # pour un calcul de surface fraction (1 grain = sa surface) ou par grain (1 grain = 1 unite)
    Anisotrop = anisotrop # pour afficher la taille moyenne ou lanisotropie = longueur max / largeur pendiculaire
    listd=[]
    lista=[]

   # on calcule les tailles moyennes quelque soit option
    listd = []
    if GRID != "Hexagon": 
        for gr in TabG :
            size = sqrt(len(gr.listpix)) * XStep * sqrt(4./math.pi)
            listd.append(size)
    else:
        for gr in TabG :
            size = sqrt(len(gr.listpix)) * XStep * (sqrt(sqrt(3) / 2)) * sqrt(4./math.pi)
            listd.append(size)

    maxi = max(listd)
    mini = min(listd)

    if dmin == 0 and dmax == 0:
        dmin = mini
        dmax = maxi
        
    if step==0: step=1
    dmin = int(floor(dmin / step)) * step
    if dmin < step: dmin = 0
    dmax = int(ceil(dmax / step)) * step

    while float(dmax - dmin) / step < 5:
        step = float(step) / 2

    if Anisotrop: # on calcule l anisotropie de tous les grains si option choisie
        EtatRun.set("\n"+"Calculation of grain anisotropy..."+"\n")
        LabelRun["fg"] = "red"
        frameFile.update()
        lenTot = len(TabG)
        stepg = max(lenTot/StProg,1)
        lista = []
        i=0
        for gr in TabG :
            if not i%stepg: UpDateProgress((i*100)/lenTot)
            if len(gr.listpix)>500: a = CalcAnisotropy(gr)
            else: a = 0
            lista.append(a)
            i+=1
        amaxi = max(lista)
        listap = copy.deepcopy(lista)
        while 0 in listap: listap.remove(0)
        amini = min(listap)
        if amin == 0 and amax ==0:
            amin = amini
            amax = amaxi
        listap = []
        UpDateProgress(0)
        EtatRun.set("\n"+"Ready."+"\n")
        LabelRun["fg"] = "dark green"
        

    listddistrib = []
    i = 0
    if not Anisotrop:
        while i <= dmax / step:
            listddistrib.append(0)
            i = i + 1
    else:
        while i <= amax / step:
            listddistrib.append(0)
            i = i + 1

    if Areafreq==0 and Anisotrop==0: # TAILLE DES GRAINS
        dmean = 0
        c = 0
        for d in listd:
            if dmin < d < dmax:
                n = int(d / step)
                listddistrib[n] = listddistrib[n] + 1 # un grain compte pour une unite qui sera normalisee par la suite
                dmean += d
                c += 1
            else:pass
        dmean = float(dmean) / c
    elif Areafreq==1 and Anisotrop==0:
        dmean = 0
        c = 0
        for d in listd:
            if dmin < d < dmax:
                n = int(d / step)
                listddistrib[n] = listddistrib[n] + d**2 # un grain compte pour sa surface qui sera normalisee par la suite
                dmean += d**2
                c += d
            else:pass
        dmean = float(dmean) / c
    elif Areafreq==0 and Anisotrop==1: # ANISOTROPIE DES GRAINS
        dmean = 0
        amean = 0
        c = 0
        for i in range(len(listd)):
            d = listd[i]
            a = lista[i]
            if amin < a < amax:
                n = int(a / step)
                listddistrib[n] = listddistrib[n] + 1 # un grain compte pour une unite qui sera normalisee par la suite
                amean += a
                c += 1
            else:pass
        amean = float(amean) / c
    else:
        dmean = 0
        amean = 0
        c = 0
        for i in range(len(listd)):
            d = listd[i]
            a = lista[i]
            if amin < a < amax:
                n = int(a / step)
                listddistrib[n] = listddistrib[n] + d**2 # un grain compte pour sa surface qui sera normalisee par la suite
                amean += a
                c += 1
            else:pass
        amean = float(amean) / c
        
    nb = reduce(add, listddistrib)
    listf = []
    for d in listddistrib: listf.append(float(d) / nb)

    if opt == "graind":t = "Daughter "
    else: t = "Parent "


    Buttonf = Button(frameHisto, text="x", height=1)
    Buttonf["background"] = "grey"
    Buttonf.bind("<Button>", CloseCVHisto)
    Buttonf.grid (row=0, column=3, ipadx=1, sticky='E')
        
    fHisto = Frame(frameHisto, borderwidth=1)
    fHisto.grid(row=1, column=0,columnspan=2,sticky='EW')
    if not Anisotrop:
        tt = t + 'Grain Sizes: ' + '\n' + '  Min = ' + cut(mini) + '  Max = ' + cut(maxi) + '   Mean = ' + cut(dmean) + ' ' + scalebar
        l = Label(fHisto, text=tt).grid(row=1, column=0, columnspan=3, sticky='EW')
##        l = Label(fHisto, text='Choose Histogram parameters').grid(row=2, column=0, columnspan=3, sticky='EW')
        l = Label(fHisto , text='< Grains <').grid(row=3, column=1, sticky='EW')
        EntryMin = Entry(fHisto, width=15)
        EntryMin.insert(0, cut(dmin))
        EntryMin.bind("<Return>", attr_histvar)
        EntryMin.grid(row=3, column=0, sticky='EW')
        EntryMax = Entry(fHisto, width=15)
        EntryMax.insert(0, cut(dmax))
        EntryMax.bind("<Return>", attr_histvar)
        EntryMax.grid(row=3, column=2, sticky='EW')
        l = Label(fHisto , text='Histogram Step').grid(row=5, column=0, sticky='EW')
        EntryStepHisto = Entry(fHisto, width=15)
        EntryStepHisto.insert(0, step)
        EntryStepHisto.bind("<Return>", attr_histvar)
        EntryStepHisto.grid(row=5, column=1, sticky='EW')
        l = Label(fHisto , text=scalebar).grid(row=5, column=2, sticky='EW')
    else:
        tt = t + 'Grain Anistropy: ' + '\n' + '  Min = ' + cut(amini) + '  Max = ' + cut(amaxi) + '   Mean = ' + cut(amean) + ' ' 
        l = Label(fHisto, text=tt + '\n' + '---------------------').grid(row=1, column=0, columnspan=3, sticky='EW')
        l = Label(fHisto, text='Choose Histogram parameters').grid(row=2, column=0, columnspan=3, sticky='EW')
        l = Label(fHisto , text='< Anisotropy <').grid(row=3, column=1, sticky='EW')
        EntryMin = Entry(fHisto, width=15)
        EntryMin.insert(0, cut(amin))
        EntryMin.bind("<Return>", attr_histvarA)
        EntryMin.grid(row=3, column=0, sticky='EW')
        EntryMax = Entry(fHisto, width=15)
        EntryMax.insert(0, cut(amax))
        EntryMax.bind("<Return>", attr_histvarA)
        EntryMax.grid(row=3, column=2, sticky='EW')
        l = Label(fHisto , text='Histogram Step').grid(row=5, column=0, sticky='EW')
        EntryStepHisto = Entry(fHisto, width=15)
        EntryStepHisto.insert(0, step)
        EntryStepHisto.bind("<Return>", attr_histvarA)
        EntryStepHisto.grid(row=5, column=1, sticky='EW')

    l = Label(fHisto , text='Graphic Size').grid(row=7, column=0, sticky='EW')
    EntryCVSize = Entry(fHisto, width=10)
    EntryCVSize.insert(0, CVSize)
    EntryCVSize.bind("<Return>", attr_histvar)
    EntryCVSize.grid(row=7, column=1, sticky='EW')
    l = Label(fHisto , text=' pixels', height=1).grid(row=7, column=2, sticky='EW')
    l = Label(fHisto , text='', height=1).grid(row=8, column=0, sticky='EW')

    l = Label(fHisto , text=' Freq. mode ', height=1,width=15).grid(row=9, column=0, sticky='EW')
    ButtonArea = Button(fHisto, text="Area Fract.", height=1, width=15)
    if Areafreq==0: ButtonArea["background"] = "grey"
    else: ButtonArea["background"] = "dark green"
    ButtonArea.bind("<Button>", AreaFreq)
    ButtonArea.grid (row=9, column=2, ipadx=1, sticky='E')
    ButtonGrain = Button(fHisto, text="Grain Freq.", height=1, width=15)
    if Areafreq==0: ButtonGrain["background"] = "dark green"
    else: ButtonGrain["background"] = "grey"
    ButtonGrain.bind("<Button>", GrainFreq)
    ButtonGrain.grid (row=9, column=1, ipadx=1, sticky='E')

    l = Label(fHisto , text=' Histogram ', height=2,width=15).grid(row=10, column=0, sticky='EW')
    ButtonMeanSize = Button(fHisto, text="Grain Sizes", height=1, width=15)
    if Anisotrop==0: ButtonMeanSize["background"] = "dark green"
    else: ButtonMeanSize["background"] = "grey"
    ButtonMeanSize.bind("<Button>", MeanSizeFreq)
    ButtonMeanSize.grid (row=10, column=1, ipadx=1, sticky='E')
    ButtonAnisotrop = Button(fHisto, text="Grain Anisot.", height=1, width=15)
    if Anisotrop==1: ButtonAnisotrop["background"] = "dark green"
    else: ButtonAnisotrop["background"] = "grey"
    ButtonAnisotrop.bind("<Button>", AnisotropFreq)
    ButtonAnisotrop.grid (row=10, column=2, ipadx=1, sticky='E')
    
    l = Label(fHisto , text='', height=2).grid(row=11, column=0, sticky='EW')

    fHisto2 = Frame(frameHisto, borderwidth=1)
    fHisto2.grid(row=10, column=1,columnspan=10,sticky='EW')
    bimage = Button(fHisto2, text="Save to image", foreground="white", background ="grey",width=15,command=saveHisto)
    bimage.grid(row=0, column=0, sticky=E + W)
    bfile = Button(fHisto2, text="Save to text file", foreground="white", background ="grey",width=15,command=exportHisto)
    bfile.grid(row=0, column=1, sticky=E + W)

    labely = "Freq.(%)"
    if not Anisotrop:
        labelx = "Size (" + scalebar + ")"
        info = labelx, labely, step, dmin, dmax
    else:
        labelx = "Ellipsoide ratio a/b"
        info = labelx, labely, step, amin, amax
    
    try: CVHisto = Histogram(frameHisto, listf, info, CVSize, 0, FileName+"GrainSize"+t)
    except: pass

#===================================================================================================
    
def HistoDisorientations(listangaxe, opt='graind', scalebar="Deg.", step=2, dmin=0, dmax=0, gradx=10, grady=5, CVSize=200):
    
    global fHisto, fHisto2, Buttonf, CVHisto, OptShowB

    if opt == 'graind':
        TM = TMD
        TS = TSD
        GroupeRot = GroupeRotD
    elif opt == 'grainp':
        TM = TMP
        TS = TSP
        GroupeRot = GroupeRotP

    try: Buttonf.destroy()
    except:pass

    try: fHisto.destroy()
    except:pass

    try: fHisto2.destroy()
    except:pass

    try: CVHisto.destroy()
    except:pass

    def showPolFigAxe(event):
    # voir CC_Histogram pour les references des origines

        x = event.x
        y = event.y
        rect = CVHisto.find_closest(x, y)
        i = string.atoi(CVHisto.gettags(rect)[0])
        listaxes = listddistribaxes[i + int(dmin / step)]
        X = i * step + dmin
        t = "For Rotation Angles of " + cut(X) + " " + " +/- " + str(step) + " " + scalebar + "\n" + "Rotation Axes in Crystal Frame"
        EtatRun.set("\n"+"Preparing figure..."+"\n")
        LabelRun["fg"] = "red"
        frameFile.update()
        PolFigDirections (root, 400, listaxes, TM, TS, GroupeRot, t, "canvas", CrystD.syst,6)
        EtatRun.set("\n"+"Done"+"\n")
        LabelRun["fg"] = "blue"
        frameFile.update()

    def OpenCVHisto(event):
        HistoDisorientations(listangaxe, opt, scalebar, step, dmin, dmax, gradx, grady, CVSize)

    def CloseCVHisto(event):
        global Buttonf, fHisto
        fHisto.destroy()
        fHisto2.destroy()
        try: CVHisto.destroy()
        except:pass
        Buttonf = Button(frameHisto, text="Disorientations \n Statistics")
        Buttonf["background"] = "grey"
        Buttonf.bind("<Button>", OpenCVHisto)
        Buttonf.grid (row=0, column=0, ipadx=1, sticky='W')
        frameHisto.update()

    def attr_histvar(event):
        #global step,dmin,dmax
        step = string.atof(EntryStepHisto.get())
        dmin = string.atof(EntryMin.get())
        dmax = string.atof(EntryMax.get())
        gradx = string.atoi(EntryGradx.get())
        grady = string.atoi(EntryGrady.get())
        CVSize = string.atoi(EntryCVSize.get())
        CVHisto.destroy()
        HistoDisorientations(listangaxe, opt, scalebar, step, dmin, dmax, gradx, grady,CVSize)

    def exportHisto():
        try:
            d = getResultsPath()+ '\Results'
            os.chdir(d)
        except:
            d = getResultsPath()
            os.chdir(d)
        filehist = open("histogram_Misorientations.txt","w")
        filehist.write("step "+str(step)+"\n")
        for f in listf:
            filehist.write(str(f*100)+"\n")
        filehist.close()
        tkMessageBox.showinfo("info", "the text file of the histogram is saved in the directory: " + d)
        os.startfile(d)

    def saveHisto():
        try:
            v = CVHisto.find_withtag("value1")
            CVHisto.delete(v)
            v = CVHisto.find_withtag("value2")
            CVHisto.delete(v)
        except: pass
        try:
            d = getResultsPath()+ '\Results'
            os.chdir(d)
        except:
            d = getResultsPath()
            os.chdir(d)

        WC,HC = CVHisto.winfo_width(), CVHisto.winfo_height()
        x1,y1 = CVHisto.winfo_rootx(), CVHisto.winfo_rooty()
        ImageGrab.grab((x1,y1,x1+WC,y1+HC)).save(filename + '.bmp')
        tkMessageBox.showinfo("info", "the image of the histogram is saved in the directory: " + d)
        os.startfile(d)

    listang = []
    for angaxe in listangaxe :
        listang.append(angaxe[0])
    #----------------------------------------

    maxi = max(listang)
    mini = min(listang)

    if dmin == 0 and dmax == 0:
        dmin = mini
        dmax = maxi

    dmin = int(floor(dmin / step)) * step
    if dmin < 1: dmin = 0
    dmax = int(ceil(dmax / step)) * step

    if step > dmax / 2:
        EtatRun.set("\n"+"Choose lower step for the histogram"+"\n")
        LabelRun["fg"] = "red"

    listddistrib = []
    listddistribaxes = []
    i = 0
    while i <= dmax / step:
        listddistrib.append(0)
        listddistribaxes.append([])
        i = i + 1

    for i in range(len(listang)):
        a = listang[i]
        if a < dmax:
            n = int(a / step)
            listddistrib[n] = listddistrib[n] + 1
            listddistribaxes[n].append(listangaxe[i][1])

    nbang = reduce(add, listddistrib)
    listf = []
    for d in listddistrib: listf.append(float(d) / nbang)

    #----------------------------------------
    if opt == "graind":t = "Daughter "
    else: t = "Parent "
    #----------------------------------------

    Buttonf = Button(frameHisto, text="x", height=1, width=2)
    Buttonf["background"] = "grey"
    Buttonf.bind("<Button>", CloseCVHisto)
    Buttonf.grid (row=0, column=0, ipadx=1, sticky='E')
    fHisto = Frame(frameHisto, borderwidth=1)
    fHisto.grid(row=1, column=0, sticky='EW')
    tt = t + 'Disorientation angles: ' + '\n' + 'Min = ' + cut(mini) + ' Max = ' + cut(maxi) + ' ' + scalebar
    l = Label(fHisto, text=tt + '\n' + '---------------------').grid(row=1, column=0, columnspan=3, sticky='EW')
    l = Label(fHisto, text='Choose Histogram Parameters:').grid(row=2, column=0, columnspan=3, sticky='EW')

    l = Label(fHisto , text='Ang Min,Max').grid(row=3, column=0, sticky='EW')
    EntryMin = Entry(fHisto, width=7)
    EntryMin.insert(0, cut(dmin))
    EntryMin.bind("<Return>", attr_histvar)
    EntryMin.grid(row=3, column=1, sticky='EW')
    EntryMax = Entry(fHisto, width=7)
    EntryMax.insert(0, cut(dmax))
    EntryMax.bind("<Return>", attr_histvar)
    EntryMax.grid(row=3, column=2, sticky='EW')
    l = Label(fHisto , text='Histogram Step').grid(row=5, column=0, sticky='EW')
    EntryStepHisto = Entry(fHisto, width=7)
    EntryStepHisto.insert(0, step)
    EntryStepHisto.bind("<Return>", attr_histvar)
    EntryStepHisto.grid(row=5, column=1, sticky='EW')
    l = Label(fHisto , text=scalebar).grid(row=5, column=2, sticky='EW')

    l = Label(fHisto , text='Scale Marks x,y').grid(row=6, column=0, sticky='EW')
    EntryGradx = Entry(fHisto, width=7)
    EntryGradx.insert(0, gradx)
    EntryGradx.bind("<Return>", attr_histvar)
    EntryGradx.grid(row=6, column=1, sticky='EW')
    EntryGrady = Entry(fHisto, width=7)
    EntryGrady.insert(0, grady)
    EntryGrady.bind("<Return>", attr_histvar)
    EntryGrady.grid(row=6, column=2, sticky='EW')

    l = Label(fHisto , text='Graphic Size').grid(row=7, column=0, sticky='EW')
    EntryCVSize = Entry(fHisto, width=7)
    EntryCVSize.insert(0, CVSize)
    EntryCVSize.bind("<Return>", attr_histvar)
    EntryCVSize.grid(row=7, column=1, sticky='EW')
    l = Label(fHisto , text=' pixels', height=1).grid(row=7, column=2, sticky='EW')
    l = Label(fHisto , text='', height=1).grid(row=8, column=0, sticky='EW')

    labelx = "Angles (" + scalebar + ")"
    labely = "Freq.(%)"
    info = labelx, labely, step, dmin, dmax, gradx, grady

    CVHisto = Histogram(frameHisto, listf, info, CVSize,0, FileName+"Disorient"+t)
    CVHisto.bind("<Double-Button-1>", showPolFigAxe)

    fHisto2 = Frame(frameHisto, borderwidth=1)
    fHisto2.grid(row=10, column=1,columnspan=10,sticky='EW')
    bimage = Button(fHisto2, text="Save to image", foreground="white", background ="grey",width=15,command=saveHisto)
    bimage.grid(row=0, column=0, sticky=E + W)
    bfile = Button(fHisto2, text="Save to text file", foreground="white", background ="grey",width=15,command=exportHisto)
    bfile.grid(row=0, column=1, sticky=E + W)

#===================================================================================================

def StatisticParentGrain(nb):
    
    global currentindgrp
    # draw the pole figure of the grains (daughter or parent)

    try:
        indgrp = TabPix[nb][4]
        #print indgrp,currentindgrp
        wd = str("current parent grain=  " + str(indgrp))
        wd = wd + "daughter grain = " + str(TabPix[nb][3])
        #if abs(i-iFinish)+abs(j-jFinish)>3:
        InfoMap.set(wd)
        E1.set(str(TabParentGrains[TabPix[nb][4] - 1].euler1)[:6])
        E2.set(str(TabParentGrains[TabPix[nb][4] - 1].euler2)[:6])
        E3.set(str(TabParentGrains[TabPix[nb][4] - 1].euler3)[:6])
        VariantNumber.set(str(TabGrains[TabPix[nb][3] - 1].variant))
        QualInd.set(str(TabGrains[TabPix[nb][3] - 1].confind))
    except:
        print "PB in StatisticsParentGrain"

    if Transition != "Sigma3":

        NbVar = len(QuotientDir)
        NbOp = len(listOpDisor)

        if indgrp == 0:
            listnbvar = [0 for i in range(NbVar)]
            listnbop = [0 for i in range(NbOp)]
            UpdateStatistic(listnbvar, listnbop, indgrp)

        elif 1: #indgrp!=currentindgrp: je ne sais plus pourquoi javais ecrit cette condition
            currentindgrp = indgrp
            grp = TabParentGrains[indgrp - 1]
            listgrd = grp.graind
            listnbvar = [0 for i in range(NbVar)]
            for indgrd in listgrd:
                grd = TabGrains[indgrd - 1]
                var = grd.variant
                if len(var) == 1:
                    listnbvar[var[0] - 1] = listnbvar[var[0] - 1] + 1
            maxi = add.reduce(listnbvar)
            listnbvar = map(lambda x: (100 * x) / maxi, listnbvar)

            listnbop = [0 for i in range(NbOp)]
            for indgrd in listgrd:
                grd = TabGrains[indgrd - 1]
                vard = grd.variant
                if len(vard) == 1:
                    nvard = vard[0]
                    listindgrdvois = grd.listgrvois
                    for indgrdvois in listindgrdvois:
                        grdvois = TabGrains[indgrdvois - 1]
                        vardvois = grdvois.variant
                        if len(vardvois) == 1:
                            nvardvois = vardvois[0]
                            j = 0
                            for opvar in listOpVar:
                                if [nvard, nvardvois] in opvar:
                                    op = j
                                    listnbop[op] = listnbop[op] + 1
                                    break
                                j = j + 1
            maxi = add.reduce(listnbop)
            listnbop = map(lambda x: (100 * x) / maxi, listnbop)

            UpdateStatistic(listnbvar, listnbop, indgrp)

    else:

        NbOp = len(listOpDisor)
        listnbop = [0 for i in range(NbOp)]
        currentindgrp = indgrp
        grp = TabParentGrains[indgrp - 1]
        listgrd = grp.graind
        for indgrd in listgrd:
            grd = TabGrains[indgrd - 1]
            numop = grd.variant[0]
            listnbop[numop] += 1
        sumop = add.reduce(listnbop)
        listnbop = map(lambda x: float(100 * x) / sumop, listnbop)
        #listnbop = listnbop [:listTwinNum[order+1]]
        UpdateStatistic([], listnbop, indgrp)


#===================================================================================================

def StatisticVariants():
    global NextStep

    if Transition != "Sigma3":

        NbVar = len(QuotientDir)
        NbOp = len(listOpDisor)
        listnbvar = [0 for i in range(NbVar)]
        listnbop = [0 for i in range(NbOp)]

        for grp in TabParentGrains :
            listgrd = grp.graind
            for indgrd in listgrd:
                grd = TabGrains[indgrd - 1]
                vard = grd.variant
                if len(vard) == 1:
                    listnbvar[vard[0] - 1] = listnbvar[vard[0] - 1] + 1
                    nvard = vard[0]
                    listindgrdvois = grd.listgrvois
                    for indgrdvois in listindgrdvois:
                        grdvois = TabGrains[indgrdvois - 1]
                        vardvois = grdvois.variant
                        if len(vardvois) == 1:
                            nvardvois = vardvois[0]
                            j = 0
                            for opvar in listOpVar:
                                if [nvard, nvardvois] in opvar:
                                    op = j
                                    listnbop[op] = listnbop[op] + 1
                                    break
                                j = j + 1

        sumvar = add.reduce(listnbvar)
        listnbvar = map(lambda x: (100 * x) / sumvar, listnbvar)
        sumop = add.reduce(listnbop)
        listnbop = map(lambda x: (100 * x) / sumop, listnbop)

        UpdateStatistic(listnbvar, listnbop, -1)

    else: #Sigma3
        NbOp = len(listOpDisor)
        listnbop = [0 for i in range(NbOp)]
        for grp in TabParentGrains :
            listgrd = grp.graind
            for indgrd in listgrd:
                grd = TabGrains[indgrd - 1]
                numop = grd.variant[0]
                listnbop[numop] = listnbop[numop] + 1

        sumop = add.reduce(listnbop)
        listnbop = map(lambda x: float(100 * x) / sumop, listnbop)
        #listnbop = listnbop [:listTwinNum[order+1]]

        UpdateStatistic([], listnbop, -1)
    NextStep += 1

#===================================================================================================

def UpdateStatistic(listnbvar, listnbop0, indgrp, titre=''):
    
    global frameStatistic, fStat2, CVStatistic, CVStatistic2, ButtonfS
    global flagT  #only for Sigma3 Transition
    global H, s2
    # indgrp = -1 => All, -10 = show also Random GB

    H, s2 = 0, 0

    if Transition == "Sigma3":
        listnbop = copy.deepcopy(listnbop0)
        l = 0
        i = 0
        for nbvar in listnbop:
            if nbvar > 0: l = i
            i += 1
        listnbop = listnbop[:l + 1]
    else: listnbop = listnbop0


    if Transition == "Sigma3":
        flagT = 0 # used to distinguish if all the operators must be drawn
        listnbopT = []

    try: ButtonfS.destroy()
    except:pass

    try: CVStatistic.destroy()
    except: pass

    try: CVStatistic2.destroy()
    except: pass

    try: fStat2.destroy()
    except: pass

    def motion1(event):
        v = CVStatistic.find_withtag("value1")
        CVStatistic.delete(v)
        v = CVStatistic.find_withtag("value2")
        CVStatistic.delete(v)
        try:
            x = event.x
            y = event.y
            rect = CVStatistic.find_closest(x, y)
            i = string.atoi(CVStatistic.gettags(rect)[0])
            f = listnbvar[i]
            Y = (H - f) * F
            X = s * i
            CVStatistic.create_rectangle(x - 30, Y - 15 + 10, x + 30, Y - 15 - 10, tags="value2", fill="white")
            ty = str(f)[:4]
            CVStatistic.create_text(x, Y - 15, text='Var. ' + str(i + 1) + ', ' + ty + '%', font="Times 8", tags="value1", fill="black")
        except:pass

    def motion2(event):
        v = CVStatistic2.find_withtag("value1")
        CVStatistic2.delete(v)
        v = CVStatistic2.find_withtag("value2")
        CVStatistic2.delete(v)
        try:
            x = event.x
            y = event.y
            rect = CVStatistic2.find_closest(x, y)
            i = string.atoi(CVStatistic2.gettags(rect)[0])
            if indgrp == -10:
                if Transition != "Sigma3" or flagT == 0: f = listnbop[i + 1]
                else: f = listnbopT[i + 1]
            else:
                if Transition != "Sigma3" or flagT == 0: f = listnbop[i]
                else: f = listnbopT[i]
            Y = (H - f) * F
            X = s2 * i
            CVStatistic2.create_rectangle(x - 40, Y - 15 + 10, x + 40, Y - 15 - 10, tags="value2", fill="white")
            try: ty = str(f)[:4]
            except: ty = str(f)
            if i == -1: wd = "Rand."
            else:
                if Transition != "Sigma3": wd = 'Op.' + str(i)
                else:
                    if i == 0: wd = "Id."
                    else:
                        if flagT: wd = "S3^" + str(i)
                        else: wd = "S3^" + listTwinNom[i][0]
            CVStatistic2.create_text(x, Y - 15, text=wd + ', ' + ty + '%)', font="Times 8", tags="value1", fill="black")
        except: pass

    def OpenCVStatistic(event):
        UpdateStatistic(listnbvar, listnbop, indgrp)

    def CloseCVStatistic(event):
        global ButtonfS
        try: CVStatistic.destroy()
        except: pass
        try: CVStatistic2.destroy()
        except: pass
        try : ButtonfS.destroy()
        except: pass
        try: ButtonAskTwin.destroy()
        except: pass
        try: fStat2.destroy()
        except: pass
        frameStatistic.update()
        ButtonfS = Button(frameStatistic, text="Variants & Operators \n Statistics")
        ButtonfS["background"] = "grey"
        ButtonfS.bind("<Button>", OpenCVStatistic)
        ButtonfS.grid (row=0, column=0, ipadx=1, sticky='W')

    def exportHisto():
        try:
            d = getResultsPath()+ '\Results'
            os.chdir(d)
        except:
            d = getResultsPath()
            os.chdir(d)
        filehist = open("histogram_Operators.txt","w")

        t2=""
        if Transition != "Sigma3" and indgrp != -1 and indgrp != -10:
            filehist.write("Percents of variants in Parent Grain " + str(indgrp)+ "\n")
            filehist.write("From 1 to "+str(len(listnbvar))+"\n")
            for f in listnbvar:
                filehist.write(str(f)+"\n")
            t2 ="s"
        filehist.write("\nPercents of operators\n")
        filehist.write("From 0 to "+str(len(listnbop))+"\n")
        for f in listnbop:
            filehist.write(str(f)+"\n")
        filehist.close()
        tkMessageBox.showinfo("info", "text file of the histogram"+t2+" saved in the directory: " + d)
        os.startfile(d)

    def saveHisto():
        try:
            v = CVStatistic2.find_withtag("value1")
            CVHisto.delete(v)
            v = CVStatistic2.find_withtag("value2")
            CVHisto.delete(v)
        except: pass
        try:
            v = CVStatistic.find_withtag("value1")
            CVHisto.delete(v)
            v = CVStatistic.find_withtag("value2")
            CVHisto.delete(v)
        except: pass
        
        try:
            d = getResultsPath()+ '\Results'
            os.chdir(d)
        except:
            d = getResultsPath()
            os.chdir(d)

        filename = FileName + "Operator_Histogram"
        WC,HC = CVStatistic2.winfo_width(), CVStatistic2.winfo_height()
        x1,y1 = CVStatistic2.winfo_rootx(), CVStatistic2.winfo_rooty()
        ImageGrab.grab((x1,y1,x1+WC,y1+HC)).save(filename + '.bmp')
        t2=""

        if Transition != "Sigma3" and indgrp != -1 and indgrp != -10:
            filename = FileName + "Variant_Histogram_PG" + str(indgrp)
            WC,HC = CVStatistic.winfo_width(), CVStatistic.winfo_height()
            x1,y1 = CVStatistic.winfo_rootx(), CVStatistic.winfo_rooty()
            ImageGrab.grab((x1,y1,x1+WC,y1+HC)).save(filename + '.bmp')
            t2="s"

        tkMessageBox.showinfo("info", "histogram"+t2+ " saved in the directory: " + d)
        os.startfile(d)
        

    F = 2 #facteur de grossissement
    M = 20 #marge
    Label(frameStatistic, text="").grid(row=0, column=1, sticky='EW')
    ButtonfS = Button(frameStatistic, text="x", height=1, width=2)
    ButtonfS["background"] = "grey"
    ButtonfS.bind("<Button>", CloseCVStatistic)
    ButtonfS.grid (row=0, column=2, ipadx=1, sticky='E')

    if Transition == "Sigma3":
        def AskTwin(event):
            global flagT
            if flagT == 0: flagT = 1
            else : flagT = 0
            if len(listnbopT) == 0:
                listnbopT.append(listnbop[0])
                for i in range(1, len(listnbop)):
                    stat = listnbop[i]
                    if i in listTwinNum:listnbopT.append(listnbop[i])
                    else : listnbopT[-1] += listnbop[i]
            if flagT: drawStat(listnbopT)
            else: drawStat(listnbop)

        def AskTwinRand(event):
            global flagT
            if flagT == 0: flagT = 1
            else : flagT = 0
            if len(listnbopT) == 0:
                listnbopT.append(listnbop[0])
                listnbopT.append(listnbop[1])
                for i in range(2, len(listnbop)):
                    stat = listnbop[i]
                    if i - 1 in listTwinNum:listnbopT.append(listnbop[i])
                    else : listnbopT[-1] += listnbop[i]
            if flagT: drawStat(listnbopT)
            else: drawStat(listnbop)

        ButtonAskTwin = Button(frameStatistic, text="Twinning orders \n or Operator details")
        ButtonAskTwin["background"] = "grey"
        if indgrp == -10: ButtonAskTwin.bind("<Button>", AskTwinRand)
        else: ButtonAskTwin.bind("<Button>", AskTwin)
        ButtonAskTwin.grid (row=0, column=0, ipadx=1, sticky='W')

    if Transition != "Sigma3" and indgrp != -1 and indgrp != -10:
        if len(listnbvar) < 15: L = 100
        else : L = 200
        s = int(L / len(listnbvar))
        H = max(listnbvar) + 2 * M #100 
        CVStatistic = Canvas(frameStatistic , width=len(listnbvar) * s * F + 2 * M, height=(H + 15) * F, borderwidth=1, highlightthickness=1)
        CVStatistic["relief"] = SOLID
        CVStatistic.grid(row=1, column=0, sticky='EW')
        #CVStatistic.delete(ALL)
        if indgrp != -1: CVStatistic.create_text(int((L * F / 2) + M), M, text=str("Variant Statistics (%)\n in Parent Grain " + str(indgrp)), font="Times 9")
        else: CVStatistic.create_text(int((L * F / 2) + M), M, text="Variant Statistics" , font="Times 10")
        i = 0
        for nbvar in listnbvar:
            if i == 0: Rectangle(CVStatistic, s * i * F + M, H * F, (s * i + s) * F + M, (H - nbvar) * F, fill="green", tags=str(i))
            else: Rectangle(CVStatistic, s * i * F + M, H * F, (s * i + s) * F + M, (H - nbvar) * F, fill="red", tags=str(i))
            CVStatistic.create_text((s * i + int(s / 2)) * F + M, (H + 10) * F, text=str(i + 1), font="Times 8 ")
            i = i + 1
        CVStatistic.bind('<Motion>', motion1)

    else: L = len(listnbop) * 10
       

    def drawStat(lno):
        global frameStatistic, fStat2,CVStatistic2, H, s2

        try: CVStatistic2.destroy()
        except: pass
        H = max(lno) + 2 * M
        s2 = int(L / len(lno))
        CVStatistic2 = Canvas(frameStatistic , width=len(lno) * s2 * F + 2 * M, height=(H + 15) * F, borderwidth=1, highlightthickness=1)
        CVStatistic2["relief"] = SOLID
        CVStatistic2.grid(row=3, column=0, sticky='EW')
        CVStatistic2.bind("<Double-ButtonPress-3>", saveHisto)
        if indgrp != -1 and indgrp != -10:
            CVStatistic2.create_text(int((L * F / 2) + M), M, text=str("Operator Statistics (%)\n in Parent Grain " + str(indgrp)), font="Times 9")
        else:
            if len(titre) == 0:
                if Transition != "Sigma3": wd = "Operator Statistics"
                else: wd = "Operator Freq. / Primary Grain"
            else: wd = titre
            CVStatistic2.create_text(int((L * F / 2) + M), M, text=wd, font="Times 10")

        if indgrp != -10:
            i = 0
            for nbvar in lno:
                if i == 0: Rectangle(CVStatistic2, s2 * i * F + M, H * F, (s2 * i + s2) * F + M, (H - nbvar) * F, fill="green", tags=str(i))
                else: Rectangle(CVStatistic2, (s2 * i) * F + M, H * F, (s2 * i + s2) * F + M, (H - nbvar) * F, fill="red", tags=str(i))
                if Transition != "Sigma3": wd = str(i)
                else:
                    if flagT: wd = str(i)
                    else: wd = listTwinNom[i][0]
                CVStatistic2.create_text((s2 * i + int(s2 / 2)) * F + M, (H + 10) * F, text=wd, font="Times 8 ")
                i = i + 1
        else:
            i = -1
            for nbvar in lno:
                if i == -1: Rectangle(CVStatistic2, s2 * (i + 1) * F + M, H * F, (s2 * (i + 1) + s2) * F + M, (H - nbvar) * F, fill="grey", tags=str(i))
                elif i == 0: Rectangle(CVStatistic2, s2 * (i + 1) * F + M, H * F, (s2 * (i + 1) + s2) * F + M, (H - nbvar) * F, fill="green", tags=str(i))
                else: Rectangle(CVStatistic2, (s2 * (i + 1)) * F + M, H * F, (s2 * (i + 1) + s2) * F + M, (H - nbvar) * F, fill="red", tags=str(i))
                if i == -1: wd = "Rand."
                else:
                    if Transition != "Sigma3": wd = str(i)
                    else:
                        if flagT: wd = str(i)
                        else: wd = listTwinNom[i][0]
                CVStatistic2.create_text((s2 * (i + 1) + int(s2 / 2)) * F + M, (H + 10) * F, text=wd, font="Times 8 ")
                i = i + 1

        CVStatistic2.bind('<Motion>', motion2)

        fStat2 = Frame(frameStatistic, borderwidth=1)
        fStat2.grid(row=30, column=0,columnspan=2,sticky='EW')
        bimage = Button(fStat2, text="Save to image", foreground="white", background ="grey",width=15,command=saveHisto)
        bimage.grid(row=0, column=0, sticky=E + W)
        bfile = Button(fStat2, text="Save to text file", foreground="white", background ="grey",width=15,command=exportHisto)
        bfile.grid(row=0, column=1, sticky=E + W)

    drawStat(listnbop)
    

#===================================================================================================

def HistoX(listf, info, texto='text', CVSize=200, opt=1, text2=''):
    
    global fHisto, Buttonf, CVHisto, lHisto

    try: Buttonf.destroy()
    except:pass
    try: fHisto.destroy()
    except:pass
    try: CVHisto.destroy()
    except:pass
    try: lHisto.destroy()
    except:pass

    def OpenCVHisto(event):
        HistoX(listf, info, texto, CVSize, 1, text2)

    def CloseCVHisto(event):
        global Buttonf, fHisto, lHisto
        fHisto.destroy()
        try: CVHisto.destroy()
        except:pass
        try: lHisto.destroy()
        except:pass
        Buttonf = Button(frameHisto, text=texto)
        Buttonf["background"] = "grey"
        Buttonf.bind("<Button>", OpenCVHisto)
        Buttonf.grid (row=0, column=0, ipadx=1, sticky='W')
        lHisto.update()
        frameHisto.update()

    Buttonf = Button(frameHisto, text="x", height=1, width=2)
    Buttonf["background"] = "grey"
    Buttonf.bind("<Button>", CloseCVHisto)
    Buttonf.grid (row=0, column=0, ipadx=1, sticky='E')
    lHisto = Label(frameHisto, text=text2)
    lHisto.grid (row=1, column=0, sticky='EW')
    fHisto = Frame(frameHisto, borderwidth=1)
    fHisto.grid(row=2, column=0, sticky='EW')

    CVHisto = Histogram(frameHisto, listf, info, CVSize, opt, FileName+"HistoX")
    
#====================================================================================================

def CalculateDisorGrains(opt="graind", resultmode="angaxe", betweenvois = 1):
    
    # etablit la table des disorientations sous forme matricielle en tous les grains voisins si betweenvois = 1,
    # et entre tous les grains fills dun meme grain parent sinon
    
    TabG = []

    if opt == "graind":
        TabG = TabGrains
        t = "daughter"
    elif opt == "grainp":
        TabG = TabParentGrains
        t = "parent"

    EtatRun.set("\n"+"Calculation of disorientations between " + t + " grains. Running..."+"\n")
    LabelRun["fg"] = "red"
    frameFile.update()

    stepg = max(len(TabG)/StProg,1)

    if betweenvois==1:
        if resultmode == "angaxe":
            listangaxe = []
            for i in range(1, len(TabG) + 1):
                if not i%stepg: UpDateProgress((i * 100) / len(TabG))
                gr = TabG[i - 1]
                listindvois = gr.listgrvois
                for j in listindvois:
                    angle, axe = disorientationgrainANGAXE(i, j,opt)
                    listangaxe.append([angle, axe])
            return listangaxe

        elif  resultmode == "mat":
            listmat = []
            for i in range(1, len(TabG) + 1):
                if not i%stepg: UpDateProgress((i * 100) / len(TabG))
                gr = TabG[i - 1]
                listindvois = gr.listgrvois
                for j in listindvois:
                    mat = disorientationgrain(i, j,opt)
                    listmat.append(mat)
            return listmat

    else:
        if resultmode == "angaxe":
            listangaxe = []
            for i in range(1, len(TabG) + 1):
                if not i%stepg: UpDateProgress((i * 100) / len(TabG))
                for j in range(i+1, len(TabG) + 1):
                    angle, axe = disorientationgrainANGAXE(i, j,opt)
                    listangaxe.append([angle, axe])
            return listangaxe

        elif  resultmode == "mat":
            listmat = []
            for i in range(1, len(TabG) + 1):
                if not i%stepg: UpDateProgress((i * 100) / len(TabG))
                for j in range(i+1, len(TabG) + 1):
                    mat = disorientationgrain(i, j,opt)
                    listmat.append(mat)
            return listmat
        

#===================================================================================================

def StatisticFreqMisD():
    StatisticFreqMis('graind',1)
    
#===================================================================================================

def StatisticFreqMisDAll():
    StatisticFreqMis('graind',0)

#===================================================================================================
def StatisticFreqMisP():
    StatisticFreqMis('grainp',1)

#===================================================================================================
def StatisticFreqMisPAll():
    StatisticFreqMis('grainp',0)
    
#===================================================================================================

def StatisticFreqMis(opt='graind',betweenvois=1):
    
    global OptShowB
    
    OptShowB = opt
    listangaxe = CalculateDisorGrains(opt, "angaxe",betweenvois)
    UpDateProgress(0)
    EtatRun.set("\n"+"Done."+"\n")
    LabelRun["fg"] = "blue"
    frameFile.update()
    HistoDisorientations(listangaxe, 'graind', "Deg.", 1, 0, 0, 5,5,200)

#===================================================================================================

def StatisticSigma3nD():
    StatisticSigma3n("graind")

#===================================================================================================

def StatisticSigma3nP():
    StatisticSigma3n("grainp")

#===================================================================================================

def StatisticSigma3n(opt="graind"):
    global OptShowB
    
    OptShowB = opt
    listdes = CalculateDisorGrains(opt, "mat")
    #print "la reconnaissance des operateurs se fait avec AngTolGrain comme tolerance angulaire"
    #print "******************************************************************************************"
    listfreqop = [0 for i in range(len(listOpDisor))] # les frequences sont mises pour tous les op. sans les classer par order de maclage
    UpDateProgress(0)
    EtatRun.set("\n"+"Identifying the misorientations to the twin operators Sigma3n, n<= " + str(OrderTwin) + " ..."+"\n")
    LabelRun["fg"] = "red"
    frameFile.update()

    lendes = len(listdes)
    stepg = max(lendes/StProg,1)
    i = 0
    for des in listdes:
        if not i%stepg: UpDateProgress((des*100)/lendes)
        nop = testS3n(des, AngTolGrain, OrderTwin)
        if nop >= 0: listfreqop[nop] += 1
        i = i + 1

    nbtot = add.reduce(listfreqop)
    listfsigma3n = []
    for d in listfreqop: listfsigma3n.append((100 * d) / nbtot)
    UpDateProgress(0)

    EtatRun.set("\n"+"Done."+"\n")
    LabelRun["fg"] = "blue"
    frameFile.update()

    UpdateStatistic([], listfsigma3n, -1)

#===================================================================================================

def calculateLLC(listch):
    # calcul de LLC = Length of Longest Chain in a list of chains

    try: LLC = max([len(ch) for ch in listch])
    except: return 0

    def LCcompoch(chi, chj):
        # chi et chj sont deux chaines et il faut calculer la taille chi-1*chj
        # cherchele point de deviation des 2 chaines
        k0 = 0
        for k in range(min(len(chi), len(chj))):
            if chi[k] == chj[k]:k0 = k + 1
            else:break
        LC = len(chi) - k0 + len(chj) - k0
        return LC

    for i in range(len(listch)):
        chi = listch[i]
        for j in range(i, len(listch)):
            chj = listch[j]
            LC = LCcompoch(chi, chj)
            if LC > LLC: LLC = LC

    return LLC

#===================================================================================================

def DistribLLC():
    # calculate the distribution of the LLCs
    listch = []
    distrib = []

    for grp in TabParentGrains:
        listchgrp = []
        listch0 = []
        for indgrd in grp.graind:
            grd = TabGrains[indgrd - 1]
            opch = grd.variant
            try: listch0.append([indgrd, stringify(opch[1])]) #opch[0] = nom de loperateur
            except : pass # il se peut que certains graind naient pas ete indexe et grd.variant=[0] dans ce cas
        for opch0 in listch0:
            flag = 1
            for i in range(len(listchgrp)): # on doit obtenir une liste de [[indgrd],ch,...]
                opch = listchgrp[i]
                if opch0[1] == opch[1]:
                    listchgrp[i][0].append(opch0[0])
                    flag = 0
                    break
            if flag: listchgrp.append([[opch0[0]], opch0[1]])
        listchs = []
        for ch in listchgrp:
            listchs.append(ch[1])
        LLC = calculateLLC(listchs)
        distrib.append(LLC)
        listch.extend(listchgrp)

    return listch, distrib

#===================================================================================================

def DrawFractalAll():
    
    global TRDTreeDone
    # draw the fractal of all the TRDs

    if not TRDTreeDone:
        TRDTrees()
        EtatRun.set("\n"+"Calculating TRD Trees..."+"\n")
        LabelRun["fg"] = "red"
        frameFile.update()

    listCH, distribLLC = DistribLLC()
    TwinTreeListOp(listCH, 0)

    EtatRun.set("\n"+"Done."+"\n")
    LabelRun["fg"] = "blue"

#===================================================================================================

def StatisticNbGinTRDs():
    # number of daughter grains in TRDs

    distrib = []

    for grp in TabParentGrains:
        nb = len(grp.graind)
        distrib.append(nb)

    listf = [0 for i in range(max(distrib) + 1)]
    nbmean = 0
    nbmax = 0
    for nb in distrib:
        listf[nb] += 1
        nbmean += nb
        if nb>nbmax : nbmax = nb

    nbmean = float(nbmean) / len(distrib)

    listf = dot(1. / len(distrib), array(listf)).tolist()

    labelx = "Nb of grains per TRD"
    labely = "Freq.(%)"
    info = labelx, labely, 1, 0, max(distrib) + 1
    m = "<Nb of Grains per TRD> = " + cut(nbmean) + ", max = " + str(nbmax)

    HistoX(listf, info, "Nb of Grains per TRD \n Statistics", 200, 1, m)

    EtatRun.set("\n"+"Done."+"\n")
    LabelRun["fg"] = "blue"

#===================================================================================================

def StatisticLLC():
    
    global TRDTreeDone
    # draw the fractal of all the TRDs

    if not TRDTreeDone:
        TRDTrees()
        EtatRun.set("\n"+"Calculating TRD Trees..."+"\n")
        LabelRun["fg"] = "red"
        frameFile.update()

    listCH, distribLLC = DistribLLC()
    
    listf = [0 for i in range(max(distribLLC) + 1)]
    llcmean = 0
    LLCmax = 0
    for llc in distribLLC:
        listf[llc] += 1
        llcmean += llc
        if llc>LLCmax :LLCmax = llc
    llcmean = float(llcmean) / len(distribLLC)

    listf = dot(1. / len(distribLLC), array(listf)).tolist()

    labelx = "LLC"
    labely = "Freq.(%)"
    info = labelx, labely, 1, 0, max(distribLLC) + 1
    titre = "LLC"
    m = "<LLC>  = " + cut(llcmean) + ", max = " + str(LLCmax)

    HistoX(listf, info, titre, 200, 1, m)

    EtatRun.set("\n"+"Done."+"\n")
    LabelRun["fg"] = "blue"
#===================================================================================================

def StatisticsFractalAll():

    def I(x): # nb de grains theorique dans une fractale d ordre n sans polysynthetisme
        I = 1 + 2*(3**x-1)
        return I

    EtatRun.set("\n"+"Calculating statistics..."+"\n")
    LabelRun["fg"] = "red"
    frameFile.update()
    pmean = 0 # polysynthetism
    amean = 0 # anistropy
    pmax = 0
    amax = 0
    NTRD = 0

    for indgrp in range(1,len(TabParentGrains)+1):
        nbcases  = 0
        nbgrains = 0
        nbcasestheor = 0
        grp = TabParentGrains[indgrp - 1]
        listch0 = []
        for indgrd in grp.graind:
            grd = TabGrains[indgrd - 1]
            opch = grd.variant
            try: listch0.append([indgrd, stringify(opch[1])]) #opch[0] = nom de loperateur
            except : pass # il se peut que certains graind naient pas ete indexe et grd.variant=[0] dans ce cas
        #print "listch0",listch0
        listch = []
        for opch0 in listch0:
            flag = 1
            for i in range(len(listch)): # on doit obtenir une liste de [[indgrd],ch,...]
                opch = listch[i]
                if opch0[1] == opch[1]:
                    listch[i][0].append(opch0[0])
                    flag = 0
                    break
            if flag: listch.append([[opch0[0]], opch0[1]])
        #print "listch",listch
        listchs = []
        for ch in listch:
            listchs.append(ch[1])
        LLC = calculateLLC(listchs)
        #TwinTreeListOp(listch, LLC)
        for ch in listch:
            nbgrains+= len(ch[0])
            nbcases+=1
        nbcasestheor+= I(float(LLC)/2)
        try:
            #p = twin polysynthetism
            p = float(nbgrains)/nbcases
            # a = twin anistropy
            a = float(nbcasestheor)/nbcases
            pmean+=p
            amean+=a
            NTRD+=1
            if p>pmax:pmax = p
            if a >amax: amax = a
        except: pass
    pmean = pmean / NTRD
    amean = amean / NTRD

    EtatRun.set("\n"+"<Twin Polysynthetism> = "+ cut(pmean) + ", max = "+ cut(pmax) +
                "    \n &    <Twin Anisotropy> = " + cut(amean) + ", max = "+ cut(amax))
    LabelRun["fg"] = "blue"

#===================================================================================================

def DrawFractal(nb):
    
    global currentindgrp
    # draw the fractal of one TRD

    EtatRun.set("\n"+"Please wait..."+"\n")
    LabelRun["fg"] = "red"
    frameFile.update()

    try:
        indgrp = TabPix[nb][4]
        #print indgrp,currentindgrp
        wd = str("current parent grain=  " + str(indgrp))
        wd = wd + "   daughter grain = " + str(TabPix[nb][3])
        InfoMap.set(wd)
        E1.set(str(TabParentGrains[TabPix[nb][4] - 1].euler1)[:6])
        E2.set(str(TabParentGrains[TabPix[nb][4] - 1].euler2)[:6])
        E3.set(str(TabParentGrains[TabPix[nb][4] - 1].euler3)[:6])
        VariantNumber.set(str(TabGrains[TabPix[nb][3] - 1].variant))
    except: pass

    if 1: # indgrp!=currentindgrp: mais je sais plus pourquoi jai ecrit ca
        currentindgrp = indgrp
        grp = TabParentGrains[indgrp - 1]
        listch0 = []
        for indgrd in grp.graind:
            grd = TabGrains[indgrd - 1]
            opch = grd.variant
            try: listch0.append([indgrd, stringify(opch[1])]) #opch[0] = nom de loperateur
            except : pass # il se peut que certains graind naient pas ete indexe et grd.variant=[0] dans ce cas
        #print "listch0",listch0
        listch = []
        for opch0 in listch0:
            flag = 1
            for i in range(len(listch)): # on doit obtenir une liste de [[indgrd],ch,...]
                opch = listch[i]
                if opch0[1] == opch[1]:
                    listch[i][0].append(opch0[0])
                    flag = 0
                    break
            if flag: listch.append([[opch0[0]], opch0[1]])
        #print "listch",listch
        listchs = []
        for ch in listch:
            listchs.append(ch[1])
        LLC = calculateLLC(listchs)
        TwinTreeListOp(listch, LLC)

    EtatRun.set("\n"+"Done."+"\n")
    LabelRun["fg"] = "blue"

#===================================================================================================

def DrawPoleParentGrain(nb):
    
    global currentindgrp
    # draw the pole figure of the grains (daughter or parent)

    EtatRun.set("\n"+"Please wait..."+"\n")
    LabelRun["fg"] = "red"
    frameFile.update()

    try:
        indgrp = TabPix[nb][4]
        #print indgrp,currentindgrp
        wd = str("current parent grain=  " + str(indgrp))
        wd = wd + "   daughter grain = " + str(TabPix[nb][3])
        #if abs(i-iFinish)+abs(j-jFinish)>3:
        InfoMap.set(wd)
        E1.set(str(TabParentGrains[TabPix[nb][4] - 1].euler1)[:6])
        E2.set(str(TabParentGrains[TabPix[nb][4] - 1].euler2)[:6])
        E3.set(str(TabParentGrains[TabPix[nb][4] - 1].euler3)[:6])
        VariantNumber.set(str(TabGrains[TabPix[nb][3] - 1].variant))
        QualInd.set(str(TabGrains[TabPix[nb][3] - 1].confind))
    except: pass

    if 1: # indgrp!=currentindgrp: mais je sais plus pourquoi jai ecrit ca
        currentindgrp = indgrp
        grp = TabParentGrains[indgrp - 1]
        listEulersP = [[grp.euler1, grp.euler2, grp.euler3]]
        listEulersD = []
        for indgrd in grp.graind:
            grd = TabGrains[indgrd - 1]
            phi1, phi, phi2 = grd.euler1, grd.euler2, grd.euler3
            listEulersD.append([phi1, phi, phi2])
        if listEulersP != [[0, 0, 0]] and Transition != "Sigma3":
            PolFigEulers (root, 400, [listEulersP, listEulersD], TMD, TSD, GroupeRotD, TMP, TSP, GroupeRotP, FileName + "_PG " + str(indgrp), "Parent+Daughter", "canvas",SourceFile)
        elif Transition == "Sigma":
            PolFigEulers (root, 400, [listEulersP, listEulersD], TMD, TSD, GroupeRotD, TMP, TSP, GroupeRotP, FileName + "_PG " + str(indgrp), "Parent+Daughter", "canvas",SourceFile)
        else: PolFigEulers (root, 400, [listEulersD], TMD, TSD, GroupeRotD, TMP, TSP, GroupeRotP, FileName + "_PG " + str(indgrp), "Daughter", "canvas", SourceFile)

    EtatRun.set("\n"+"Done."+"\n")
    LabelRun["fg"] = "blue"
        # car 0 = mauvaise indexation

#===================================================================================================

def DrawPoleParentGrain2(nb):
    
    global currentindgrp
    # draw the pole figure of the pixels (daughter or parent)

    EtatRun.set("\n"+"Please wait..."+"\n")
    LabelRun["fg"] = "red"
    frameFile.update()

    try:
        indgrp = TabPix[nb][4]
        #print indgrp,currentindgrp
        wd = str("current parent grain=  " + str(indgrp))
        wd = wd + "   daughter grain = " + str(TabPix[nb][3])
        #if abs(i-iFinish)+abs(j-jFinish)>3:
        InfoMap.set(wd)
        E1.set(str(TabParentGrains[TabPix[nb][4] - 1].euler1)[:6])
        E2.set(str(TabParentGrains[TabPix[nb][4] - 1].euler2)[:6])
        E3.set(str(TabParentGrains[TabPix[nb][4] - 1].euler3)[:6])
        VariantNumber.set(str(TabGrains[TabPix[nb][3] - 1].variant))
        QualInd.set(str(TabGrains[TabPix[nb][3] - 1].confind))
    except: pass


    if indgrp != currentindgrp:
        currentindgrp = indgrp
        grp = TabParentGrains[indgrp - 1]
        listEulersP = [[grp.euler1, grp.euler2, grp.euler3]]
        listEulersD = []
        for indgrd in grp.graind:
            grd = TabGrains[indgrd - 1]
            for indpix in grd.listpix:
                phi1, phi, phi2 = TabPix[indpix][0], TabPix[indpix][1], TabPix[indpix][2]
                listEulersD.append([phi1, phi, phi2])
        mode = "draw"
        #mode = "canvas"
        if listEulersP != [[0, 0, 0]]: PolFigEulers (root, 400, [listEulersP, listEulersD], TMD, TSD, GroupeRotD, TMP, TSP, GroupeRotP, FileName + "_PG" + str(indgrp), "Parent+Daughter2", mode, SourceFile)
        else: PolFigEulers (root, 400, [listEulersD], TMD, TSD, GroupeRotD, TMP, TSP, GroupeRotP, FileName + "_PGpix" + str(indgrp), "Daughter2", mode,SourceFile)

    EtatRun.set("\n"+"Done."+"\n")
    LabelRun["fg"] = "blue"
        # car 0 = mauvaise indexation    

#===================================================================================================

def DrawPoleFig(phase):
    # draw the pole figure of the grains (daughter or parent)
    mode = "canvas"
    if phase == "Daughter":
        title = FileName + "_DG"
        TabG = TabGrains
    elif phase == "Parent":
        title = FileName + "_PG"
        TabG = TabParentGrains
    elif phase == "Daughter Pixels":
        title = FileName + "_DPix"
        mode="draw"
    elif phase == "Daughter Pixels Density":
        title = FileName + "_DPix Density"
        mode = "iso"
    elif phase == "Daughter Density":
        TabG = TabGrains
        title = FileName + "_D Density"
        mode = "iso"
    else: print "PB in DrawPoleFig"
    
    EtatRun.set("\n"+"Please wait..."+"\n")
    LabelRun["fg"] = "red"
    frameFile.update()

    listEulers = []
    if phase == "Daughter" or phase == "Parent" or phase == "Daughter Density":
        for gr in TabG:
            phi1, phi, phi2 = gr.euler1, gr.euler2, gr.euler3
            if phi1 != 0 and phi != 0 and phi2 != 0: listEulers.append([phi1, phi, phi2]) # car 0 = mauvaise indexation

    elif phase == "Daughter Pixels" or phase == "Daughter Pixels Density":
        for pix in TabPix:
            phi1, phi, phi2 = pix[:3]
            if phi1 != 0 and phi != 0 and phi2 != 0: listEulers.append(pix[:3]) # car 0 = mauvaise indexation
       
    PolFigEulers (root, 400, [listEulers], TMD, TSD, GroupeRotD, TMP, TSP, GroupeRotP, title, phase,mode, SourceFile)
    #oct 2017
    #PolFigEulers (frameLivePF, 400, [listEulers], TMD, TSD, GroupeRotD, TMP, TSP, GroupeRotP, title, phase,mode)
    frameLivePF.update()    

    EtatRun.set("\n"+"Done."+"\n")
    LabelRun["fg"] = "blue"
    
#============================================================================================================================
                   # CONTROLE DES MOUVEMENTS DE LA SOURIS SUR LE CANVAS 
#============================================================================================================================

def ijFromxy(x, y):
    j = int(x / Scale)
    if GRID != "Hexagon": i = int(y / Scale)
    else: i = int((y / (Scale * (sqrt(3) / 2))))
    return i, j

#====================================================================================================
def Displacexy(x, y):

    slx = float(scrx.get()[0]) * int(canvas["width"]) # x displacment due to scroll
    sly = float(scry.get()[0]) * int(canvas["height"]) # y displacment due to scroll

    xn = int(x + (1. / (float(scrx.get()[1]) - float(scrx.get()[0]))) * slx)
    yn = int(y + (1. / (float(scry.get()[1]) - float(scry.get()[0]))) * sly)

    return xn, yn
#============================================================================================================================

def motion(event):
    global canvas

    rect = canvas.find_withtag("selection")
    canvas.delete(rect)

    HPhkl = canvas.find_withtag("HPhklr")
    canvas.delete(HPhkl)

    HPhkl = canvas.find_withtag("HPhkl")
    canvas.delete(HPhkl)
    
    xs = event.x
    ys = event.y  
    xs, ys = Displacexy(xs, ys)

    hp = canvas.find_closest(xs, ys,1)
    label = canvas.gettags(hp)
    if "HP" in label:
        canvas.create_rectangle(xs - 30, ys - 15 + 10, xs+30, ys - 15 - 10, tags="HPhklr", fill="white")
        l = label[1]
        l=l.replace("[","(")
        l=l.replace("]",")")
        canvas.create_text(xs, ys - 15, text=l, font="Times 8", tags="HPhkl", fill="black")
        
    
    i, j = ijFromxy(xs, ys)
    nb = i * Nbpixx + j
    if OptShowB == "pixels":
        try:
            if TabPix[nb][5]!=300:
                InfoMap.set(str("current daughter pixel=  " + str(nb)))
                E1.set(cut(TabPix[nb][0]))
                E2.set(cut(TabPix[nb][1]))
                E3.set(cut(TabPix[nb][2]))
                VariantNumber.set("")
                QualInd.set(cut(TabPix[nb][5]))
            else:
                pixP = TabPixP[nb]
                if pixP[0]!=0: InfoMap.set("Retained Parent")
                else: InfoMap.set("Unindexed pixel")
                E1.set(cut(pixP[0])) # 3 = graind
                E2.set(cut(pixP[1]))
                E3.set(cut(pixP[2]))
        except: pass

        try:
            m_CtoS = EulersToMat(TabPix[nb][0], TabPix[nb][1], TabPix[nb][2],SourceFile) #transf matrix crystal ref cubique to sample
            m_CtoS = dot(inv(TSD), m_CtoS) #transf matrix crystal to sample
            CX = dot(m_CtoS,array([1,0,0]))
            CY = dot(m_CtoS,array([0,1,0]))
            CZ = dot(m_CtoS,array([0,0,1]))
            #print "Z axis in crystal basis = ", CZ
            axeX = IntegriseAxe(CX)
            axeY = IntegriseAxe(CY)
            axeZ = IntegriseAxe(CZ)
            EX.set(str(axeX))
            EY.set(str(axeY))
            EZ.set(str(axeZ))
        except: pass


        
    elif OptShowB == "pixelsP":
        try:
            InfoMap.set(str("current parent pixel=  " + str(nb)))
            E1.set(cut(TabPixP[nb][0]))
            E2.set(cut(TabPixP[nb][1]))
            E3.set(cut(TabPixP[nb][2]))
            VariantNumber.set("")
            QualInd.set("")
        except: pass
    elif OptShowB == "graind":
        try:
            indgrd = TabPix[nb][3]
            if indgrd:
                wd = str("daughter grain=  " + str(indgrd)) # 3 = graind
                indgrp = TabPix[nb][4]
                if indgrp: wd = wd + "\n" + "parent grain = " + str(indgrp) #4 = grainp
                else: wd = wd + "\n" + " unindexed parent grain "  #4 = grainp = 0
                InfoMap.set(wd)
                grd = TabGrains[indgrd - 1]
                #print " grains voisins = ", grd.listgrvois
                E1.set(cut(grd.euler1)) # 3 = graind
                E2.set(cut(grd.euler2))
                E3.set(cut(grd.euler3))
                QualInd.set(cut(grd.confind))
                if Transition == "Sigma3":
                    try: wd = str(listTwinNom[grd.variant[0]][0]) + ", '" + stringify(grd.variant[1]) + "'"
                    except: wd = ''
                else: wd = str(grd.variant)
                VariantNumber.set(wd)
            else:
                if TabPix[nb][5]==300:
                    pixP = TabPixP[nb]
                    InfoMap.set("Retained Parent")
                    E1.set(cut(pixP[0])) # 3 = graind
                    E2.set(cut(pixP[1]))
                    E3.set(cut(pixP[2]))
                else:
                    InfoMap.set("Lost Pixel")
                    E1.set("0") # 3 = graind
                    E2.set("0")
                    E3.set("0")
                VariantNumber.set(" ")
                QualInd.set("")
        except: pass

    elif OptShowB == "grainp" or OptShowB == "statistic":
        try:
            indgrp = TabPix[nb][4]
            retflag = (TabPix[nb][5]==300)
            if indgrp == 0 and not retflag:
                    InfoMap.set("unindexed parent grain")
                    E1.set("0") # 3 = graind
                    E2.set("0")
                    E3.set("0")
                    QualInd.set("")
                    VariantNumber.set(" ")             
            elif indgrp and not retflag:
                wd = str("Reconstructed parent grain=  " + str(indgrp))
                indgrd = TabPix[nb][3]
                grp = TabParentGrains[indgrp - 1]
                #print " grains voisins = ", grp.listgrvois
                wd = wd + "\n" + "daughter grain = " + str(indgrd)
                InfoMap.set(wd)
                E1.set(cut(grp.euler1))
                E2.set(cut(grp.euler2))
                E3.set(cut(grp.euler3))
                QualInd.set(cut(grp.confind))
                if indgrd:
                    gr = TabGrains[indgrd - 1]
                    if Transition == "Sigma3":
                        try: wd = str(listTwinNom[gr.variant[0]][0]) + ", '" + stringify(gr.variant[1]) + "'"
                        except: wd = ''
                    else: wd = str(gr.variant)
                    VariantNumber.set(wd)
                else:
                    InfoMap.set("unindexed")
                    E1.set("0") # 3 = graind
                    E2.set("0")
                    E3.set("0")
                    QualInd.set("")
                    VariantNumber.set(" ")
            else:
                if retflag: #retained parent
                    indgrret = abs(TabPixP[nb][4])
                    grpret = TabRetainedGrains[indgrret - 1]
                    #print " grains voisins = ", grpret.listgrvois
                    wd =  str("Retained parent grain= " + str(indgrret))
                    InfoMap.set(wd)
                    E1.set(cut(grpret.euler1))
                    E2.set(cut(grpret.euler2))
                    E3.set(cut(grpret.euler3))
                    QualInd.set(cut(grpret.confind))
                else: pass
        except: pass

#====================================================================================================
        
def press(event):
    global canvas

    rect = canvas.find_withtag("selection")
    canvas.delete(rect)

    xs = event.x
    ys = event.y
            
    xs, ys = Displacexy(xs, ys)
    i, j = ijFromxy(xs, ys)
    nb = i * Nbpixx + j
    
    g = 0
    if OptShowB == "pixels":
        try: g = TabPix[nb][3] # 3 = graind
        except: pass
    elif OptShowB == "graind":
        try: g = TabGrains[TabPix[nb][3] - 1].graind # 3 = graind
        except: pass
    elif OptShowB == "grainp":
        try: g = TabParentGrains[TabPix[nb][4] - 1].grainp # 4 = grainp
        except: pass
    elif OptShowB == "polefig": DrawPoleParentGrain(nb)
    elif OptShowB == "polefig2": DrawPoleParentGrain2(nb)
    elif OptShowB == "statistic": StatisticParentGrain(nb)
    elif OptShowB == "fractal": DrawFractal(nb)


#====================================================================================================    
def selectzoneStart(event):
    
    global xStart, yStart,HPCurrent, HPCurrentCol, canvas

    xs0 = event.x
    ys0 = event.y

    xStart, yStart = Displacexy(xs0, ys0)

    if HPModeMethod.get()==3: HPCurrentCol = "yellow"

    if HPModeMethod.get()==4:
        time.sleep(0.02)
        d=1
        list_hp = canvas.find_overlapping(xStart-d, yStart-d, xStart+d, yStart+d)
        for hp in list_hp:
            label = canvas.gettags(hp)
            if "HP" in label or "HPhkl" in label or "HPhklr" in label:
                canvas.delete(hp)
            #HPCurrentCol=="yellow"
            #canvas.update()

    if HPModeMethod.get()==5:
        xs, ys = Displacexy(xs0, ys0)
        hp = canvas.find_closest(xs, ys,1)
        label = canvas.gettags(hp)
        if "HP" in label:
            canvas.create_rectangle(xs - 30, ys - 15 + 10, xs+30, ys - 15 - 10, tags="wHP", fill="white")
            l = label[1]
            l=l.replace("[","(")
            l=l.replace("]",")")
            canvas.create_text(xs, ys - 15, text=l, font="Times 8", tags="wHP", fill="black")
            
    if (HPModeMethod.get()==1 or HPModeMethod.get()==2 or HPModeMethod.get()==3 or HPModeMethod.get()==4) and HPCurrentCol=="yellow":
        d=3
        list_hp = canvas.find_overlapping(xStart-d, yStart-d, xStart+d, yStart+d) 
        for hp in list_hp:
            label = canvas.gettags(hp)
            if "HP" in label:
                HPCurrent = hp
                hpinfo = canvas.itemconfig(hp)
                # je pense que itemcget aurait donne la meme chose
                HPCurrentCol = hpinfo["fill"][-1]
                canvas.itemconfig(hp, fill="yellow")
                break
        
    elif HPModeMethod.get()==1 and HPCurrentCol!="yellow":
        try:
            coords = canvas.coords(HPCurrent)
            x1,y1,x2,y2 = coords
            xm,ym = (x2+x1)/2,(y2+y1)/2
            dx,dy = xs0-xm,ys0-ym
            x1n, y1n = Displacexy(x1+dx, y1+dy)
            x2n, y2n = Displacexy(x2+dx, y2+dy)
            canvas.coords(HPCurrent,x1n,y1n,x2n,y2n)
            #rattribuer ici la couleur originale
            canvas.itemconfig(HPCurrent, fill=HPCurrentCol)
        except: pass
        HPCurrentCol = "yellow"

    elif HPModeMethod.get()==2 and HPCurrentCol!="yellow":
        coords = canvas.coords(HPCurrent)
        x1,y1,x2,y2 = coords
        xm,ym = (x2+x1)/2,(y2+y1)/2
        dx,dy = xs0-xm,ys0-ym
        x1n, y1n = Displacexy(x1+dx, y1+dy)
        x2n, y2n = Displacexy(x2+dx, y2+dy)
        hpinfo = canvas.itemconfig(HPCurrent)
        width = canvas.itemcget(HPCurrent, "width")
        HPTrace = canvas.create_line(x1n,y1n,x2n,y2n,fill=HPCurrentCol, width = width)
        tag = canvas.gettags(HPCurrent)
        canvas.itemconfig(HPTrace, tags=tag)
        canvas.itemconfig(HPCurrent, fill=HPCurrentCol)
        #rattribuer ici la couleur originale
        HPCurrentCol = "yellow"

#====================================================================================================

def presskey(event):
    
    global canvas, HPCurrentCol
    
    sc=1  
    pkey = str(event.char)
    #print pkey
    if HPModeMethod.get()==3 and (pkey =="+" or pkey=="-"):
        if pkey =="-": sc = 0.8
        elif pkey =="+": sc = 1.2
        coords = canvas.coords(HPCurrent)
        x1,y1,x2,y2 = coords
        xm,ym = (x2+x1)/2,(y2+y1)/2
        dx,dy = x2-x1,y2-y1
        dx,dy = sc*dx,sc*dy
        x1n,x2n,y1n,y2n = xm-dx/2,xm+dx/2,ym-dy/2,ym+dy/2
        canvas.coords(HPCurrent,x1n,y1n,x2n,y2n)
        #rattribuer ici la couleur originale
        canvas.itemconfig(HPCurrent, fill=HPCurrentCol)

    elif HPModeMethod.get()==3 and (pkey =="r" or pkey=="g"or pkey=="b" or pkey=="y" or pkey=="m"):
        if pkey =="r": col = "red"
        elif pkey =="g": col = "green"
        elif pkey =="b": col = "blue"
        elif pkey =="y": col = "yellow"
        elif pkey =="p": col = "pink"
        elif pkey =="w": col = "white"
        try: canvas.itemconfig(HPCurrent, fill=col)
        except: pass
    
#====================================================================================================
def drawrectangle(event):
    
    global canvas, HPCurrent

    x = event.x
    y = event.y

    x, y = Displacexy(x, y)
         
    rect = canvas.find_withtag("selection")
    canvas.delete(rect)
    if OptShowC == "Line":
        canvas.create_line(xStart, yStart, x, y, tags="selection")
    elif OptShowC == "Rectangle":
        canvas.create_rectangle(xStart, yStart, x, y, tags="selection")                
  
#====================================================================================================
def selectzoneFinish(event):
    
    global xFinish, yFinish

    xFinish = event.x
    yFinish = event.y

    xFinish, yFinish = Displacexy(xFinish, yFinish)

    iStart, jStart = ijFromxy(xStart, yStart)
    iFinish, jFinish = ijFromxy(xFinish, yFinish)

    if OptShowC == "Line":
        nbStart = iStart * Nbpixx + jStart
        nbFinish = iFinish * Nbpixx + jFinish
        if OptShowB == "pixels":
            desMIN = disorientationpixelANGAXE(nbStart, nbFinish)
            des = disorientationpixel(nbStart, nbFinish)
            t = test(des, AngTolGrain)
            angle = desMIN[0]
            axe = IntegriseAxe(desMIN[1])
            if angle: text1 = str("mis. = " + str(angle)[:4] + " Deg / [" + str(axe[0]) + " " + str(axe[1]) + " " + str(axe[2]) + "]")
            else: text1 = str("mis. = " + str(angle)[:4] + " Deg ")
            if t:
                if Transition != "Sigma3": text = text1 + "\n" + str("=  Op. " + str(t))
                else:
                    lsigma = [str(listTwinNom[x][0]) for x in t]
                    text = text1 + "\n" + str("=  Sigma3_exp" + str(lsigma)[1:-1])
            else:
                text = text1
                
        elif OptShowB == "pixelsP":
            desMIN = disorientationpixelANGAXEP(nbStart, nbFinish)
            des = disorientationpixelP(nbStart, nbFinish)
            t = test(des, AngTolGrain)
            angle = desMIN[0]
            axe = IntegriseAxe(desMIN[1])
            if angle: text1 = str("mis. = " + str(angle)[:4] + " Deg / [" + str(axe[0]) + " " + str(axe[1]) + " " + str(axe[2]) + "]")
            else: text1 = str("mis. = " + str(angle)[:4] + " Deg")
                                           
            if t:
                if Transition != "Sigma3": text = text1 + "\n" + str("=  Op. " + str(t))
                else:
                    lsigma = [str(listTwinNom[x][0]) for x in t]
                    text = text1 + "\n" + str("=  Sigma3_exp" + str(lsigma)[1:-1])
            else:
                text = text1

        elif OptShowB == "graind":
            indgStart = TabPix[nbStart][3] # 3 = graind
            indgFinish = TabPix[nbFinish][3]
            desMIN = disorientationgrainANGAXE(indgStart, indgFinish,"graind")
            des = disorientationgrain(indgStart, indgFinish,"graind")
            t = test(des, AngTolGrain)
            angle = desMIN[0]
            axe = IntegriseAxe(desMIN[1])
            if angle: text1 = str("mis. = " + str(angle)[:4] + " Deg / [" + str(axe[0]) + " " + str(axe[1]) + " " + str(axe[2]) + "]")
            else: text1 = str("mis. = " + str(angle)[:4] + " Deg")
            if t:
                if Transition != "Sigma3": text = text1 + "\n" + str("=  Op. " + str(t))
                else:
                    lsigma = [str(listTwinNom[x][0]) for x in t]
                    text = text1 + "\n" + str("=  Sigma3_exp" + str(lsigma)[1:-1])
            else:
                text = text1

        elif OptShowB == "grainp": #ici ici desmin
            try:
                indgStart = TabPix[nbStart][4] # 4 = grainp
                indgFinish = TabPix[nbFinish][4]
                desMIN = disorientationgrainANGAXE(indgStart, indgFinish,"grainp")
                #des = disorientationgrain(indgStart, indgFinish)
                if 1:
                    angle = desMIN[0]
                    axe = IntegriseAxe(desMIN[1])
                    if angle: text = str("mis. = " + str(angle)[:4] + " Deg / [" + str(axe[0]) + " " + str(axe[1]) + " " + str(axe[2]) + "]")
                    else : text = str("mis. = "+ str(angle)[:4] + " Deg")
            except: pass

        else:
            text = ""

        if GRID != "Hexagon": vect_S = array([(iFinish - iStart), (jFinish - jStart), 0])# S = sample
        else: vect_S = array([(sqrt(3) / 2) * (iFinish - iStart), (jFinish - jStart), 0])

        if OptShowB == "graind":
            TabG = TabGrains
            TS = TSD
        elif OptShowB == "grainp" :
            TabG = TabParentGrains
            TS = TSP
        if OptShowB == "graind" or OptShowB == "grainp" :
            try:
                mStart_CtoS = EulersToMat(TabG[indgStart - 1].euler1, TabG[indgStart - 1].euler2, TabG[indgStart - 1].euler3,SourceFile) #transf matrix crystal ref cubique to sample
                mStart_CtoS = dot(inv(TS), mStart_CtoS) #transf matrix crystal to sample
                vectStart_C = IntegriseAxe(dot(mStart_CtoS, vect_S))
                text = text + "\n" + "Line = " + str(vectStart_C)
            except:
                pass

        InfoMap.set(text)

    elif OptShowC == "Rectangle":
        create_SubSet()

##    t0 = time.clock()
##    t=0
##    while t<20:
##        t = (time.clock()-t0)
##        #if abs(x-xFinish)+abs(y-yFinish)>20:
##    rect = canvas.find_withtag("selection")
##    canvas.delete(rect)
##    print "je suis sorti", time.clock()
##
#====================================================================================================
def SynchronTabPix():
    global TabPix
    i=0
    for pix in TabPix:
        if pix[5]==300:
            pix[4]=TabPixP[i][4] #indice negatif marquant retained parent (and not reconstructed parent)
        i+=1
#====================================================================================================
def ActNextStep(event):
    global NextStep
    ActStep()

def ActSameStep(event):
    global NextStep
    if NextStep > 2: NextStep -= 1
    ActStep()

def ActStep():
    global NextStep

    if Transition != "Sigma3":
        if NextStep == 0:
            EtatRun.set("\n"+"Go to Menu: PhaseTransition/Load Transition"+"\n")
            LabelRun["fg"] = "blue"
        elif NextStep == 1:
            EtatRun.set("\n"+"Go to Menu: Files/Import File"+"\n")
            LabelRun["fg"] = "blue"
        elif NextStep == 2:
            remspikes()
            if FlagSupPix.get(): dilate()
        elif NextStep == 3:IdentifyDaughterGrains()
        elif NextStep == 4:
            pass
            #IgnoreSmallGrainsD() # deja dans IdentifyDaughterGrains
        elif NextStep == 5:
            if RecMethod.get()==0: ReconstParentGrainsOrient3()
            elif RecMethod.get()==1: ReconstParentGrainsOrient4()
            elif RecMethod.get()==2: ReconstParentGrainsOrient2()
        elif NextStep == 6:IgnoreMisorientedGrainsP()
        elif NextStep == 7:
            if FlagPolish.get(): RemoveSpinesRecurs()
        elif NextStep == 8:
            if FlagHoles: RemWhiteHolesRecurs()
        elif NextStep == 9:
            AbsoluteVarNum()
            SynchronTabPix()
        elif NextStep == 10:StatisticVariants()
        elif NextStep == 11:StatisticLLC()
        elif NextStep == 12:NextStep = 5

    else:
        if NextStep == 0:
            EtatRun.set("\n"+"Go to Menu: PhaseTransition/Load Transition"+"\n")
            LabelRun["fg"] = "blue"
        elif NextStep == 1:
            EtatRun.set("\n"+"Go to Menu: Files/Import File"+"\n")
            LabelRun["fg"] = "blue"
        elif NextStep == 2:
            remspikes()
            if FlagSupPix.get():dilate()
        elif NextStep == 3:IdentifyDaughterGrains()
        elif NextStep == 4:IgnoreSmallGrainsD()
        elif NextStep == 5:ReconstTRD(OrderTwin)
        elif NextStep == 6:TRDTrees()
        elif NextStep == 7:StatisticVariants()
        elif NextStep == 8:StatisticNbGinTRDs()
        elif NextStep == 9:StatisticVariants()
        elif NextStep == 10:NextStep = 5

#====================================================================================================

def AllStepsInOne(event):
    
    global PercentRec, AngTolParentNucl, AngTolParentGrowth, PercentOrient, AngTolGrain
    global NextStep, ComptNucl, ComptGrowth, NbMinGrd

    if NextStep < 5:
        remspikes()
        if FlagSupPix.get(): dilate()
        IdentifyDaughterGrains()
        #IgnoreSmallGrainsD()
    if Transition != "Sigma3":
        while PercentRec < PercentRecLimit:
            print "Percentage of Reconstructed Grains ", PercentRec , "%"
            if RecMethod.get()==0: ReconstParentGrainsOrient3()
            elif RecMethod.get()==1: ReconstParentGrainsOrient4()
            elif RecMethod.get()==2: ReconstParentGrainsOrient2()
            AngTolParentNucl+= AngTolParentNuclStep #degree
            AngTolParentGrowth+= AngTolParentGrowthStep  #degree
            EtatAngTolParentNucl.set(AngTolParentNucl)
            EtatAngTolParentGrowth.set(AngTolParentGrowth)
            ComptNucl, ComptGrowth = 0, 0
            if AngTolParentNucl > AngTolParentNuclLimit or AngTolParentGrowth > AngTolParentGrowthLimit: 
                if not RecMethod.get(): break
                else: # refait une derniere boucle avec NbMinGrd = 3
                    NbMinGrd = 3
                    RecMethod.set(0)
            ComptNucl, ComptGrowth = 0, 0
        print "End of reconstruction. Start of orientation calculation."
        AngTolGrain = 7.0 #degree
        EtatAngTolGrain.set(AngTolGrain)
        IgnoreMisorientedGrainsP()
        IgnoreMisorientedGrainsP()
        if FlagPolish.get():
            RemoveSpinesRecurs()
            RemoveSpinesRecurs()
            RemoveSpinesRecurs() # A FINIR il faut faire une routine de recherche de morceaux isoles
        if FlagHoles.get(): RemWhiteHolesRecurs()
        IgnoreMisorientedGrainsP()
        AbsoluteVarNum()
        SynchronTabPix()
        ParentGB.set(1)
        showGBP()
        if Transition == "Fe-NW" or Transition == "Fe-KS" or Transition == "Fe-GT" : DrawSigma3Aust()
        StatisticVariants()
    else:
        ReconstTRD(OrderTwin)
        TRDTrees()
        ParentGB.set(1)
        showGBP()
        #DrawGBSigma3n()
        StatisticVariants()
        #StatisticNbGinTRDs()
        StatisticVariants()

    if FlagAutoSave.get():
        EtatRun.set("\n"+"Saving the results..."+"\n")
        LabelRun["fg"] = "red"
        SaveResults0(False,False)
        EtatRun.set("\n"+"Results saved."+"\n")
        LabelRun["fg"] = "blue"
    else: SaveResults0(True, True)

    NextStep = 5
    if Transition != "Sigma3":
        wd = "End. "+ "\n"+str(PercentRec)+ " % of Grains Reconstructed. Quality Index= " + cut(IndexQual) + " Deg." 
        EtatRun.set(wd+"\n")
    else: EtatRun.set("\n"+"End of the TRD reconstruction."+"\n")
    LabelRun["fg"] = "blue"
    try: winsound.PlaySound("!", winsound.SND_ALIAS)
    except: pass

#====================================================================================================

def AutoParamRec():
    
    global PercentRecLimit, AngTolParentNuclStep, AngTolParentGrowthStep, AngTolGBStraight,AngTolParentNuclLimit,AngTolParentGrowthLimit
    global FlagSupPix, NbMinGrd,FlagAutoSave

    def attr_param(event):
        global PercentRecLimit, NbMinGrd, AngTolParentNuclStep, AngTolParentGrowthStep, AngTolGBStraight
        global AngTolParentNuclLimit,AngTolParentGrowthLimit

        param[0] = string.atoi(entry_param[0].get())
        param[1] = string.atof(entry_param[1].get())
        for i in range(2,6):
            param[i] = string.atof(entry_param[i].get())

        NbMinGrd, AngTolParentNuclStep, AngTolParentGrowthStep, PercentRecLimit, AngTolParentNuclLimit, AngTolParentGrowthLimit = param

    fpar = Toplevel(root)
    lt = Label(fpar, foreground="black")
    lt["text"] = "Daughter Phase"
    lt.grid(row=0, column=1, sticky=W)
    Radiobutton (fpar, text="Majority ", variable=DPhaseMethod, value=0).grid (row=0, column=2, sticky=W)
    Radiobutton (fpar, text="Minority ", variable=DPhaseMethod, value=1).grid (row=0, column=3, sticky=W)
    lt = Label(fpar, foreground="black")
    lt["text"] = "Suppress unindexed pixels ?"
    lt.grid(row=1, column=1, sticky=W)
    Checkbutton (fpar, text="", variable=FlagSupPix).grid (row=1, column=2, sticky=W)
    lt = Label(fpar, foreground="black")
    lt["text"] = "Reconstruction method"
    lt.grid(row=2, column=1, sticky=W)
    Radiobutton (fpar, text="Doublets", variable=RecMethod, value=2).grid (row=2, column=2, sticky=W)
    Radiobutton (fpar, text="Triplets", variable=RecMethod, value=0).grid (row=2, column=3, sticky=W)
    Radiobutton (fpar, text="Quadruplets", variable=RecMethod, value=1).grid (row=2, column=4, sticky=W)
    lt = Label(fpar, foreground="black")
    lt["text"] = "Polish Parent Grain Boundaries ?"
    lt.grid(row=3, column=1, sticky=W)
    Checkbutton (fpar, text="", variable=FlagPolish).grid (row=3, column=2, sticky=W)
    lt = Label(fpar, foreground="black")
    lt["text"] = "Remove White Holes ?"
    lt.grid(row=4, column=1, sticky=W)
    Checkbutton (fpar, text="", variable=FlagHoles).grid (row=4, column=2, sticky=W)
    
    lt = Label(fpar, foreground="black")
    lt["text"] = "Min Nb of Daugter Grains per Parent Grains   "
    lt.grid(row=5, column=1, sticky=W)
    lt = Label(fpar, foreground="black")
    lt["text"] = "Step for the Nucleation (Deg.)"
    lt.grid(row=6, column=1, sticky=W)
    lt = Label(fpar, foreground="black")
    lt["text"] = "Step for the Growth (Deg.)"
    lt.grid(row=7, column=1, sticky=W)
    lt = Label(fpar, foreground="black")
    lt["text"] = "Stop when Rec. Percentage(%) > "
    lt.grid(row=8, column=1, sticky=W)
    lt = Label(fpar, foreground="black")
    lt["text"] = "Stop when Nucleation Angle (deg.) >"
    lt.grid(row=9, column=1, sticky=W)
    lt = Label(fpar, foreground="black")
    lt["text"] = "Stop when Growth Angle (deg.) >"
    lt.grid(row=10, column=1, sticky=W)
    
    lt = Label(fpar, foreground="black")
    lt["text"] = "AutoSave the Results ?"
    lt.grid(row=11, column=1, sticky=W)
    Checkbutton (fpar, text="", variable=FlagAutoSave).grid (row=11, column=2, sticky=W)

    lt = Label(fpar, foreground="black")
    lt["text"] = "Image Saving Mode"
    lt.grid(row=12, column=1, sticky=W)
    Radiobutton (fpar, text="Screen-Copy", variable=ImageSaveMode, value=1).grid (row=12, column=2, sticky=W)
    Radiobutton (fpar, text="Export", variable=ImageSaveMode, value=0).grid (row=12, column=3, sticky=W)

    param = [NbMinGrd, AngTolParentNuclStep, AngTolParentGrowthStep, PercentRecLimit, AngTolParentNuclLimit, AngTolParentGrowthLimit] 
    entry_param = [None for i in range(6)]
    for i in range(6):
        entry_param[i] = Entry(fpar, width=10, foreground="blue")
        if i!=0: entry_param[i].insert(0, round(param[i], 5))
        else: entry_param[i].insert(0, param[i])
        entry_param[i].bind("<Return>", attr_param)
        entry_param[i].grid(row=i + 5, column=2, sticky=E + W)


#====================================================================================================

def AutoParamGB():
    
    global AngTolGBStraight

    def attr_param(event):
        global AngTolGBStraight
        AngTolGBStraight = string.atof(entry_param.get())

    fpar = Toplevel(root)
    lt = Label(fpar, foreground="black")
    lt["text"] = ""
    lt.grid(row=12, column=1, sticky=W)
    lt = Label(fpar, foreground="black")
    lt["text"] = "Colors for the Grain Boundaries = Operators"
    lt.grid(row=13, column=1, sticky=W)

    def showcv():
        try: cv.destroy()
        except: pass
        cv = Canvas(fpar)
        cv.grid(row=15, column=1, columnspan=2, sticky=E + W)
        if len(listOpDisor) > 8: cv["height"] = 400
        elif len(listOpDisor) ==1: cv["height"] = 50
        else: cv["height"] = 150
        cv["width"] = 50
        dx, dy = 10, 10
        colors = COLORS_OP
        for n in range(len(listOpDisor) + 1):
            x1, y1, x2, y2 = dx, 15 * n + dy, 100 + dx, 15 * n + dy
            cv.create_line(x1, y1, x2, y2, width=2, fill=colors[n])
            if not n:
                t = "            Random"
            else:
                t = "            Op. " + str(n - 1)
            cv.create_text(x2 + 10, y2, text=t)

    showcv()
    lt = Label(fpar, foreground="black")
    lt["text"] = "Tolerance for straight GBs (Deg)"
    lt.grid(row=20, column=1, sticky=W)

    entry_param = Entry(fpar, width=10, foreground="blue")
    entry_param.insert(0, round(AngTolGBStraight, 5))
    entry_param.bind("<Return>", attr_param)
    entry_param.grid(row=20, column=2, sticky=E + W)

#====================================================================================================

def AutoParamEdit():
    
    global SBSize, NumSizeBig, NumSizeSmall 

    def attr_param(event):
        global SBSize, NumSizeBig, NumSizeSmall
        for i in range(0,3):
            param[i] = string.atoi(entry_param[i].get())
        SBSize, NumSizeSmall, NumSizeBig = param

    fpar = Toplevel(root)
    
    lt = Label(fpar, foreground="black")
    lt["text"] = "ScaleBar Size"
    lt.grid(row=1, column=1, sticky=W)
    lt = Label(fpar, foreground="black")
    lt["text"] = "Daughter Num. Size"
    lt.grid(row=2, column=1, sticky=W)
    lt = Label(fpar, foreground="black")
    lt["text"] = "Parent Num. Size"
    lt.grid(row=3, column=1, sticky=W)

    param = [SBSize, NumSizeSmall, NumSizeBig]
    entry_param = [None for i in range(3)]
    for i in range(0,3):
        entry_param[i] = Entry(fpar, width=10, foreground="dark green")
        entry_param[i].insert(0, int(param[i]))
        entry_param[i].bind("<Return>", attr_param)
        entry_param[i].grid(row=i+1, column=2, sticky=E + W)

#====================================================================================================

def AutoParamImage():
    
    fpar = Toplevel(root)
    lt = Label(fpar, foreground="black")
    lt["text"] = "Image Saving Mode"
    lt.grid(row=12, column=1, sticky=W)
    Radiobutton (fpar, text="Screen-Copy", variable=ImageSaveMode, value=1).grid (row=12, column=2, sticky=W)
    Radiobutton (fpar, text="Export", variable=ImageSaveMode, value=0).grid (row=12, column=3, sticky=W)

#====================================================================================================

def AutoParamSigma3n():
    
    global OrderTwin, AngTolGBStraight
    global FlagSupPix

    def attr_param(event):
        global OrderTwin, AngTolGBStraight
        OrderTwin = string.atoi(entry_param.get())
        AngTolGBStraight = string.atof(entry_paramAng.get())

    fpar = Toplevel(root)
    lt = Label(fpar, foreground="black")
    lt["text"] = "Suppress unindexed pixels?"
    lt.grid(row=0, column=1, sticky=W)
    Checkbutton (fpar, text="", variable=FlagSupPix).grid (row=0, column=2, sticky=W)
    lt = Label(fpar, foreground="black")
    lt["text"] = "Limit Twin Order"
    lt.grid(row=1, column=1, sticky=W)

    entry_param = Entry(fpar, width=10, foreground="blue")
    entry_param.insert(0, OrderTwin)
    entry_param.bind("<Return>", attr_param)
    entry_param.grid(row=1, column=2, sticky=E + W)

    lt = Label(fpar, foreground="black")
    lt["text"] = " \n Sigma3_exp(n) boundaries \n for different twinning orders: \n Distinguish polar operators?"
    lt.grid(row=2, column=1, columnspan=2, sticky=E + W)

    def showcv():
        try: cv.destroy()
        except: pass
        cv = Canvas(fpar)
        cv.grid(row=7, column=1, columnspan=2, sticky=E + W)
        cv["height"] = 120
        cv["width"] = 50
        dx, dy = 10, 10
        if not TwinFlag_pol.get() : colors = COLORS_AMB
        else : colors = COLORS_POL
        for n in range(len(colors)):
            x1, y1, x2, y2 = dx, 10 * n + dy, 100 + dx, 10 * n + dy
            cv.create_line(x1, y1, x2, y2, width=2, fill=colors[n])
            if not n: t = "            Random"
            elif n >= 5 and TwinFlag_pol.get():
                if mod(n, 2): t = str(4 + (n - 5) / 2) + "+"
                else: t = str(4 + (n - 5) / 2) + "-"
            else: t = str(n - 1)
            cv.create_text(x2 + 10, y2, text=t)

    Radiobutton (fpar, text="No", variable=TwinFlag_pol, value=0, command=showcv).grid (row=5, column=1, sticky=E + W)
    Radiobutton (fpar, text="Yes", variable=TwinFlag_pol, value=1, command=showcv).grid (row=5, column=2, sticky=E + W)
    showcv()

    lt = Label(fpar, foreground="black")
    lt["text"] = "Tolerance for straight GBs (Deg)"
    lt.grid(row=10, column=1, sticky=W)

    entry_paramAng = Entry(fpar, width=10, foreground="blue")
    entry_paramAng.insert(0, AngTolGBStraight)
    entry_paramAng.bind("<Return>", attr_param)
    entry_paramAng.grid(row=10, column=2, sticky=E + W)

#====================================================================================================

Palette = palette()

root = Tk()

FlagSupPix = IntVar()
FlagSupPix.set(1)
FlagAutoSave= IntVar()
FlagAutoSave.set(0)

TwinFlag_pol = IntVar()
TwinFlag_pol.set(0)
RecMethod = IntVar()
RecMethod.set(0) # part avec la methode des triplets par default
ImageSaveMode = IntVar()
ImageSaveMode.set(0) # 0 for screen-copy and 1 for exportcanvas
DPhaseMethod = IntVar()
DPhaseMethod.set(0) # part en supposant que Daughter Phase = Majority phase
HPPhaseMethod = IntVar()
HPPhaseMethod.set(0) # to plot the traces of the habit planes HPs 0=Daughter only, 1 = Parent only, 2 = both, 3 = both that are parallel
HPdColorMethod = IntVar()
HPdColorMethod.set(0) # color used to plot the traces of the habit planes HPs of the daughter phase
HPpColorMethod = IntVar()
HPpColorMethod.set(0) # color used to plot the traces of the habit planes HPs of the parent phase
HPModeMethod = IntVar()
HPModeMethod.set(0) # method used to move/delete the traces of the habit planes HPs
HPCurrent = 0
HPCurrentCol = "yellow"
FlagPolish = IntVar()
FlagPolish.set(0)
FlagHoles = IntVar()
FlagHoles.set(1)

WMAX = root.winfo_screenwidth()
HMAX = root.winfo_screenheight()

print "screen width  =", WMAX
print "screen height =", HMAX

#root.geometry("%dx%d+0+0" % (SCRW, SCRH))
root.title("ARPGE 2.7: Reconstruction of Parent Grains from EBSD data (Sept. 2021)")

EtatRun = StringVar()
EtatRun.set("\n"+"idle"+"\n")
EtatTrans = StringVar()
EtatTrans.set("?")
EtatFile = StringVar()
EtatFile.set("?")
EtatSize = StringVar()
EtatSize.set("?")
EtatAngTolGrain = DoubleVar()
EtatAngTolGrain.set(AngTolGrain)
EtatAngTolParentGrowth = DoubleVar()
EtatAngTolParentGrowth.set(AngTolParentGrowth)
EtatAngTolParentNucl = DoubleVar()
EtatAngTolParentNucl.set(AngTolParentNucl)
EtatMinGrainSize = IntVar()
EtatMinGrainSize.set(MinGrainSize)
EMin = DoubleVar()
EMin.set(str(0.))
EMax = DoubleVar()
EMax.set(str(180.))
DaughterGB = IntVar()
DaughterGB.set(0)
ParentGB = IntVar()
ParentGB.set(0)
VarNum = IntVar()
VarNum.set(0)
GreyFlag = IntVar()
GreyFlag.set(0)
HideFlag = IntVar()
HideFlag.set(0)

#**************************************************************************************
frameWidth = Frame(root, height=1)
#frameWidth["relief"]=SOLID
l = Label(frameWidth, text=" ", width=115)
l.grid (row=0, column=1)
frameWidth.grid (row=0, column=1, columnspan=2, sticky='W')

frameProgress = Frame(root, width=300, height=1)
frameProgress["borderwidth"] = 2
#frameProgress["relief"]=RAISED
iconDirectory = getFullPath("Icons")
os.chdir (iconDirectory)
imcircle = Image.open("circle.gif")
imcircle = imcircle .resize((15, 15))
imcircle = ImageTk.PhotoImage(imcircle)
bcircle = Button(frameProgress, image=imcircle)
bcircle .bind("<Button>", ActSameStep)
#bcircle.bind("<Double-Button-1>",AllRecInOne)
bcircle .grid(row=0, column=2, sticky='W')
imright = Image.open("arrowright.gif")
imright = imright.resize((15, 15))
imright = ImageTk.PhotoImage(imright)
bright = Button(frameProgress, image=imright)
bright.bind("<Button>", ActNextStep)
bright.grid(row=0, column=3, sticky='W')
imrightd = Image.open("arrowrightdouble.gif")
imrightd = imrightd.resize((15, 15))
imrightd = ImageTk.PhotoImage(imrightd)
brightd = Button(frameProgress, image=imrightd)
brightd.bind("<Button>", AllStepsInOne)
brightd.grid(row=0, column=4, sticky='W')
l = Label(frameProgress, text=" State : ")
l.grid (row=0, column=5, ipadx=5, sticky='W')
LabelRun = Label(frameProgress, textvariable=EtatRun)
LabelRun.grid (row=0, column=6, ipadx=5, sticky='W')
CVProgress = Canvas(frameProgress, width=200, height=5, borderwidth=1, highlightthickness=1)
CVProgress["relief"] = SOLID
CVProgress.grid(row=0, column=7, sticky='W')
frameProgress.grid (row=1, column=1, columnspan=2, ipadx=10, sticky="W")

#**************************************************************************************

def UpDateProgress(p):
    global CVProgress
    if p == 0: CVProgress.delete(ALL)
    else:
        p2 = 2 * p + 1
        Rectangle(CVProgress, 0, 0, p2, 10, fill="green")
    CVProgress.update()

#**************************************************************************************

frameHigh = Frame(root, borderwidth=2, width=300)
frameHigh["relief"] = RAISED

def OpenFrameHigh(event):
    global frameHigh, EtatListOp, frameFile, frameLivePF
    global ButtonPF
    frameHigh.destroy()
    frameHigh = Frame(root, borderwidth=2, width=300)
    frameHigh["relief"] = RAISED

    barre = ""
    barreInfo = "_________________________  Parameters _________________________ "
    frameFile = Frame(frameHigh, borderwidth=2)
    frameFile.grid (row=0, column=1, ipadx=10, sticky="W")
    l = Label(frameFile, text=barreInfo)
    l.grid (row=1, column=0, columnspan=5, sticky='EW')

    Label01 = Label(frameFile, text="Phase transition = ", height=1)
    Label01.grid (row=3, column=0, sticky='W')
    LabelTrans = Label(frameFile, textvariable=EtatTrans, height=1)
    LabelTrans.grid (row=3, column=1, columnspan=4, sticky='EW')

    Label02 = Label(frameFile, text="File name = ", height=1)
    Label02.grid (row=4, column=0, sticky='W')
    LabelFile = Label(frameFile, textvariable=EtatFile, height=1)
    LabelFile.grid (row=4, column=1, columnspan=4, sticky='EW')

    Label03 = Label(frameFile, text="File size = ", height=1)
    Label03.grid (row=5, column=0, sticky='W')
    LabelFile = Label(frameFile, textvariable=EtatSize, height=1)
    LabelFile.grid (row=5, column=1, columnspan=4, sticky='EW')

    def decrTolGrain(event):
        global AngTolGrain
        if AngTolGrain <= 1: AngTolGrain = float(AngTolGrain) / 2
        else: AngTolGrain = AngTolGrain - 1
        EtatAngTolGrain.set(AngTolGrain)

    def incrTolGrain(event):
        global AngTolGrain
        if abs(AngTolGrain - 1) < 0.0001: AngTolGrain = 1
        if AngTolGrain < 1: AngTolGrain = AngTolGrain * 2
        else: AngTolGrain = AngTolGrain + 1
        EtatAngTolGrain.set(AngTolGrain)

    def decrTolParent(event):
        global AngTolParentGrowth, ComptGrowth
        ComptGrowth = 0
        AngTolParentGrowth = AngTolParentGrowth - 0.5
        EtatAngTolParentGrowth.set(AngTolParentGrowth)

    def incrTolParent(event):
        global AngTolParentGrowth, ComptGrowth
        ComptGrowth = 0
        AngTolParentGrowth = AngTolParentGrowth + 0.5
        EtatAngTolParentGrowth.set(AngTolParentGrowth)

    def decrTolParentNucl(event):
        global AngTolParentNucl, ComptNucl
        ComptNucl = 0
        if AngTolParentNucl <= 1: AngTolParentNucl = float(AngTolParentNucl) - 0.1
        else: AngTolParentNucl = AngTolParentNucl - 0.5
        EtatAngTolParentNucl.set(AngTolParentNucl)

    def incrTolParentNucl(event):
        global AngTolParentNucl, ComptNucl
        ComptNucl = 0
        if abs(AngTolParentNucl) < 0.1: AngTolParentNucl = 1
        if AngTolParentNucl < 1: AngTolParentNucl = AngTolParentNucl + 0.1
        else: AngTolParentNucl = AngTolParentNucl + 0.5
        EtatAngTolParentNucl.set(AngTolParentNucl)

    def decrMinGrainSize(event):
        global MinGrainSize
        if MinGrainSize == 1:pass
        elif MinGrainSize <= 10: MinGrainSize = MinGrainSize - 1
        else: MinGrainSize = MinGrainSize - 5
        EtatMinGrainSize.set(MinGrainSize)

    def incrMinGrainSize(event):
        global MinGrainSize
        if MinGrainSize < 10: MinGrainSize = MinGrainSize + 1
        else: MinGrainSize = MinGrainSize + 5
        EtatMinGrainSize.set(MinGrainSize)

    def decrMinGrainSize2(event):
        global MinGrainSize
        if MinGrainSize < 1: MinGrainSize = 1
        elif MinGrainSize < 10: MinGrainSize = MinGrainSize - 1
        elif MinGrainSize <= 100: MinGrainSize = MinGrainSize - 10
        else: MinGrainSize = MinGrainSize - 50
        EtatMinGrainSize.set(MinGrainSize)

    def incrMinGrainSize2(event):
        global MinGrainSize
        if MinGrainSize > 100: MinGrainSize = MinGrainSize + 50
        elif MinGrainSize > 10: MinGrainSize = MinGrainSize + 10
        else: MinGrainSize = MinGrainSize + 1
        EtatMinGrainSize.set(MinGrainSize)

    def ChgPF(event):
        global OptShowB, ButtonPF
        ModePoleParentGrain()     

    Label1 = Label(frameFile, text="Tolerance for Grain Identification (Deg) = ", height=1)
    Label1.grid (row=6, column=0, sticky='W')
    Button1 = Button(frameFile, text="-", height=1, width=2)
    Button1.bind("<Button>", decrTolGrain)
    Button1.grid (row=6, column=2, ipadx=1, sticky='E')
    LabelAngTolGrain = Label(frameFile, textvariable=EtatAngTolGrain, height=1)
    LabelAngTolGrain.grid (row=6, column=3, ipadx=1, sticky='EW')
    Button2 = Button(frameFile, text="+", height=1, width=2)
    Button2.bind("<Button>", incrTolGrain)
    Button2.grid (row=6, column=4, ipadx=1, sticky='W')

    Label2 = Label(frameFile, text="Tolerance for Parent Grain Nucleation (Deg) = ", height=1)
    Label2.grid (row=7, column=0, sticky='W')
    Button12 = Button(frameFile, text="-", height=1, width=2)
    Button12.bind("<Button>", decrTolParentNucl)
    Button12.grid (row=7, column=2, ipadx=1, sticky='E')
    LabelAngTolParent = Label(frameFile, textvariable=EtatAngTolParentNucl, height=1)
    LabelAngTolParent.grid (row=7, column=3, ipadx=1, sticky='EW')
    Button22 = Button(frameFile, text="+", height=1, width=2)
    Button22.bind("<Button>", incrTolParentNucl)
    Button22.grid (row=7, column=4, ipadx=1, sticky='W')

    Label2 = Label(frameFile, text="Tolerance for Parent Grain Growth (Deg) = ", height=1)
    Label2.grid (row=8, column=0, sticky='W')
    Button12 = Button(frameFile, text="-", height=1, width=2)
    Button12.bind("<Button>", decrTolParent)
    Button12.grid (row=8, column=2, ipadx=1, sticky='E')
    LabelAngTolParent = Label(frameFile, textvariable=EtatAngTolParentGrowth, height=1)
    LabelAngTolParent.grid (row=8, column=3, ipadx=1, sticky='EW')
    Button22 = Button(frameFile, text="+", height=1, width=2)
    Button22.bind("<Button>", incrTolParent)
    Button22.grid (row=8, column=4, ipadx=1, sticky='W')

    Label3 = Label(frameFile, text="Minimum Accepted Grain Size (Pixels) = ", height=1)
    Label3.grid (row=9, column=0, sticky='W')
    Button13 = Button(frameFile, text="-", height=1, width=2)
    Button13.bind("<Button>", decrMinGrainSize)
    Button13.bind("<Double-Button-1>", decrMinGrainSize2)
    Button13.grid (row=9, column=2, ipadx=1, sticky='E')
    LabelAngTolParent = Label(frameFile, textvariable=EtatMinGrainSize, height=1)
    LabelAngTolParent.grid (row=9, column=3, ipadx=1, sticky='EW')
    Button23 = Button(frameFile, text="+", height=1, width=2)
    Button23.bind("<Button>", incrMinGrainSize)
    Button23.bind("<Double-Button-1>", incrMinGrainSize2)
    Button23.grid (row=9, column=4, ipadx=1, sticky='W')

    frame1 = Frame(frameHigh, borderwidth=4)
    l = Label(frame1, text=" ", width=11)
    l.grid (row=0, column=1)
    frame1.grid (row=0, column=2, sticky="W")
    frame2 = Frame(frameHigh, width=300, borderwidth=4)
    l = Label(frame2, text="_________________ Operators ________________")
    l.grid (row=0, column=1)
    EtatListOp = Listbox(frame2, width=35)
    EtatListOp.grid (row=1, column=1)
    frame2.grid (row=0, column=3, sticky="W")
    frameLivePF = Frame(frameHigh, width=300, borderwidth=4)

    ButtonPF = Button(frameLivePF, text="Parent Pole Figure\n Double-click on a parent grain", height=2, width=25,foreground="black", background="grey")
    ButtonPF.bind("<Button>", ChgPF)
    ButtonPF.grid (row=0, column=1, ipadx=1, sticky='W')
    frameLivePF.grid (row=0, column=4, sticky="W")
    # FINIR

    s = Scrollbar(frame2)
    s.grid(row=1, column=0, sticky="NS")
    s.config(command=EtatListOp.yview)
    EtatListOp.config(yscrollcommand=s.set)

    for i in range(len(listOpDisor)):
        angle, axe = listOpDisor[i][0][0], listOpDisor[i][0][1]
        #listOpDisor[i] = [[ang(min),axe,listmat],[ang,axe,listmat],etc.] 
        if angle!=0:
            axeres = SimpleAxe(axe, GroupeRotD)
            axeres= IntegriseAxe(array(axeres))
            text = str("Op. " + str(i) + " = " + str(angle)[:4] + " Deg /  [" + str(axeres[0]) + " " + str(axeres[1]) + " " + str(axeres[2]) + "]")
        else: text = str("Op. 0 = 0 Deg " )
        EtatListOp.insert(END, text)

    ButtonFrameHigh = Button(frameHigh, text="x", height=1, width=2)
    ButtonFrameHigh["background"] = "grey"
    ButtonFrameHigh.bind("<Button>", CloseFrameHigh)
    ButtonFrameHigh.grid (row=0, column=5, ipadx=1, sticky='N')
    frameHigh.grid(row=2, column=1, columnspan=2, sticky=W)

def CloseFrameHigh(event):
    global frameHigh
    frameHigh.destroy()
    frameHigh = Frame(root, borderwidth=2)
    frameHigh["relief"] = RAISED
    l = Label(frameHigh, text=" ___________Parameters __________")
    l.grid (row=0, column=0, ipadx=5)
    ButtonFrameHigh = Button(frameHigh, text="open")
    ButtonFrameHigh["background"] = "grey"
    ButtonFrameHigh.bind("<Button>", OpenFrameHigh)
    ButtonFrameHigh.grid (row=0, column=1, ipadx=1, sticky='N')
    frameHigh.grid(row=2, column=1, columnspan=2, sticky='W')

frameHigh.grid(row=2, column=1, columnspan=2, sticky='W')
OpenFrameHigh(1)

#**************************************************************************************

def magnify(event):
    global Scale, canvas
    Scale = Scale * (4./3)
    canvas.delete(ALL)
    if Scale > 1 : magnifymap()
    else:
        if not HideFlag.get(): showmap()

def demagnify(event):
    global Scale, canvas
    Scale = float(Scale) *(3./4)
    canvas.delete(ALL)
    if Scale > 1:Scale = Scale
    if Scale < 0.00001:Scale = 1
    magnifymap()

def showEuler1(event):
    global OptShowA
    refreshButtons1()
    ButtonA["foreground"] = "white"
    ButtonA["background"] = "blue"
    OptShowA = "euler1"
    if not HideFlag.get(): showmap()

def showEuler2(event):
    global OptShowA
    refreshButtons1()
    OptShowA = "euler2"
    ButtonB["foreground"] = "white"
    ButtonB["background"] = "blue"
    if not HideFlag.get(): showmap()

def showEuler3(event):
    global OptShowA
    refreshButtons1()
    OptShowA = "euler3"
    ButtonC["foreground"] = "white"
    ButtonC["background"] = "blue"
    if not HideFlag.get(): showmap()

def showEuler123(event):
    global OptShowA
    refreshButtons1()
    OptShowA = "euler123"
    ButtonD["foreground"] = "white"
    ButtonD["background"] = "blue"
    if not HideFlag.get(): showmap()

def showIPF(event):
    global OptShowA, modeIPF
    modeIPF = (modeIPF+1)%3
    refreshButtons1()
    OptShowA = "IPF"
    ButtonG["foreground"] = "white"
    ButtonG["background"] = "blue"
    if not HideFlag.get(): showmap()

def showQual(event):
    global OptShowA
    refreshButtons1()
    OptShowA = "qual"
    ButtonF["foreground"] = "white"
    ButtonF["background"] = "blue"
    if not HideFlag.get(): showmap()

def showNumbers(event):
    global OptShowA, OptShowN
    refreshButtons1()
    ButtonE["foreground"] = "white"
    ButtonE["background"] = "blue"
    if OptShowA != "numbers":
        OptShowA = "numbers"
        if not HideFlag.get(): showmap()
    OptShowN = (OptShowN+1)%2
    if OptShowN == 0: hideGrainNum()
    else: showGrainNum()
    

def showpix(event):
    global OptShowB, OptShowA, FlagPixP
    FlagPixP = 0 # pour pixels filles
    OptShowB = "pixels" # pixels of daughter grains
    if OptShowA == "numbers":
        OptShowA = "euler1"
        ButtonE["foreground"] = "black"
        ButtonE["background"] = "grey"
        ButtonA["foreground"] = "white"
        ButtonA["background"] = "blue"
    refreshButtons2()
    ButtonL["foreground"] = "white"
    ButtonL["background"] = "blue"
    if not HideFlag.get(): showmap()

def showpixP(event):
    global OptShowB, OptShowA, FlagPixP, TabPixP
    FlagPixP = 1 # pour pixels parent
    if TabPixP:
        OptShowB = "pixelsP"
    else:
        wd = "No Parent Pixel Map. Calculating Parent Pixels..." 
        EtatRun.set("\n"+wd+"\n")
        LabelRun["fg"] = "red"
        ButtonNN["foreground"] = "black"
        ButtonNN["background"] = "grey"
        calculateTabPixP()
    if OptShowA == "numbers":
        OptShowA = "euler1"
        ButtonE["foreground"] = "black"
        ButtonE["background"] = "grey"
        ButtonA["foreground"] = "white"
        ButtonA["background"] = "blue"
    refreshButtons2()
    ButtonNN["foreground"] = "white"
    ButtonNN["background"] = "blue"
    if not HideFlag.get(): showmap()
    
def showgraind(event):
    global OptShowB
    refreshButtons2()
    OptShowB = "graind"
    ButtonM["foreground"] = "white"
    ButtonM["background"] = "blue"
    if not HideFlag.get(): showmap()

def showgrainp(event):
    global OptShowB,OptShowRet
    refreshButtons2()
    OptShowB = "grainp"
    OptShowRet = OptShowRet+1
    OptShowRet= OptShowRet%3
    ButtonN["foreground"] = "white"
    ButtonN["background"] = "blue"
    if not HideFlag.get(): showmap()

def refreshButtons1():
    global listButtons1
    for b in listButtons1:
        b["foreground"] = "black"
        b["background"] = "grey"

def refreshButtons2():
    global listButtons2
    for b in listButtons2:
        b["foreground"] = "black"
        b["background"] = "grey"

def DrawPoleFigDPix():

    DrawPoleFig("Daughter Pixels")

def DrawPoleFigDPixDens():

    DrawPoleFig("Daughter Pixels Density")

def DrawPoleFigDDens():

    DrawPoleFig("Daughter Density")
    
def DrawPoleFigD():

    DrawPoleFig("Daughter")

def DrawPoleFigP():

    DrawPoleFig("Parent")

def User_Help():
    info = "Manual in preparation"
    tkMessageBox.showinfo("Help", info)

def User_Contact():
    info = "cyril.cayron@epfl.ch"
    tkMessageBox.showinfo("Contact", info)

def User_Refer():
    info1 = " * General Theory: C. Cayron, Acta Cryst. A62 (2006), 21-40. \n * Theory for sigma3n multiple twinning: C. Cayron, Acta Cryst. A63 (2007), 11-29. \n"
    info2 = " * Method: C. Cayron, B. Artaud & L. Briottet, Mater. Charact. 57 (2006) 386-401. \n * Program for parent reconstruction: C. Cayron, J. Appl. Cryst. 40 (2007) 1183-1188.\n"
    info3 = " * Program for sigma3n multiple twinning: C. Cayron, Acta Mater. 59 (2011) 252-262.\n * Variant grouping and OR colouring in steels: C. Cayron, Mater. Charac. 94 (2014) 93-110.\n"
    tkMessageBox.showinfo("References", info1+info2+info3)

#**************************************************************************************

frame3 = Frame(root, borderwidth=2)
frame3["relief"] = RAISED

frameMenuCV = Frame(frame3)
l = Label(frameMenuCV, text=" ")
l.grid (row=0, column=1, ipady=10)
ButtonA = Button(frameMenuCV, text="Euler1", width=7, height=1, foreground="white", background="blue")
ButtonA.bind("<Button>", showEuler1)
ButtonA.grid (row=1, column=2,ipadx=5, ipady=5)
ButtonB = Button(frameMenuCV, text="Euler2", width=7, height=1, foreground="black", background="grey")
ButtonB.bind("<Button>", showEuler2)
ButtonB.grid (row=1, column=3, ipadx=5, ipady=5)
ButtonC = Button(frameMenuCV, text="Euler3", width=7, height=1, foreground="black", background="grey")
ButtonC.bind("<Button>", showEuler3)
ButtonC.grid (row=1, column=4, ipadx=5, ipady=5)
ButtonD = Button(frameMenuCV, text="Euler123", width=7, height=1, foreground="black", background="grey")
ButtonD.bind("<Button>", showEuler123)
ButtonD.grid (row=1, column=5, ipadx=5, ipady=5)
ButtonG = Button(frameMenuCV, text="IPF+", width=7, height=1, foreground="black", background="grey")
ButtonG.bind("<Button>", showIPF)
ButtonG.grid (row=1, column=6, ipadx=5, ipady=5)
ButtonF = Button(frameMenuCV, text="BC/Quality", width=7, height=1, foreground="black", background="grey")
ButtonF.bind("<Button>", showQual)
ButtonF.grid (row=1, column=7, ipadx=5, ipady=5)
ButtonE = Button(frameMenuCV, text="Grain Nb.", width=7, height=1, foreground="black", background="grey")
ButtonE.bind("<Button>", showNumbers)
ButtonE.grid (row=1, column=8, ipadx=5, ipady=5)

l = Label(frameMenuCV, text=" ", width=3)
l.grid (row=2, column=8, ipady=5)

ButtonDef = Button(frameMenuCV, text="KAM+", width=7, height=1, foreground="black", background="wheat")
ButtonDef.bind("<Button>", ShowMapDef)
ButtonDef.grid (row=0, column=2, ipadx=5, ipady=5)

ButtonGOS = Button(frameMenuCV, text="GOS", width=7, height=1, foreground="black", background="wheat")
ButtonGOS.bind("<Button>", ShowMapGOS)
ButtonGOS.grid (row=0, column=3, ipadx=5, ipady=5)

CheckButtonGBDaughter = Checkbutton(frameMenuCV, text="Daughter GBs", height=1, variable=DaughterGB, command=showGBD)
CheckButtonGBDaughter.grid (row=0, column=9, ipadx=5, sticky="W")
CheckButtonGBParent = Checkbutton(frameMenuCV, text="Parent GBs", height=1, variable=ParentGB, command=showGBP)
CheckButtonGBParent.grid (row=0, column=10, ipadx=5, sticky="W")
CheckButtonGBParent = Checkbutton(frameMenuCV, text="Variant Index", height=1, variable=VarNum, command=showVarNum)
CheckButtonGBParent.grid (row=0, column=11, ipadx=5, sticky="W")

ButtonGrey = Checkbutton(frameMenuCV, text="Grey Background", height=1, variable=GreyFlag, command=showGrey)
ButtonGrey.grid (row=1, column=9, ipadx=5, sticky="W")
ButtonGrey = Checkbutton(frameMenuCV, text="Hide Map", height=1, variable=HideFlag, command=hideMap)
ButtonGrey.grid (row=1, column=10, ipadx=5, sticky="W")

MagnifBox = Frame(frameMenuCV, borderwidth=4, width=5)
immagp = Image.open("loupep.gif")
immagp = immagp.resize((20, 20))
immagp = ImageTk.PhotoImage(immagp)
immagm = Image.open("loupem.gif")
immagm = immagm.resize((20, 20))
immagm = ImageTk.PhotoImage(immagm)
LabelXY = Label(MagnifBox, text="Mag. ", height=1)
LabelXY.grid (row=0, column=1, ipady=5)
ButtonX = Button(MagnifBox, image=immagm, height=20, width=20)
ButtonX.bind("<Button>", demagnify)
ButtonX.grid (row=0, column=2, ipadx=5)
ButtonY = Button(MagnifBox, image=immagp,height=20, width=20)
ButtonY.bind("<Button>", magnify)
ButtonY.grid (row=0, column=3, ipadx=5)
MagnifBox.grid(row=1, column=11, columnspan=2, ipadx=5, ipady=5)

frameMenuCV.grid(row=1, column=1)

frameMenuCV2 = Frame(frame3)
l = Label(frameMenuCV2, text="Exp. Daughter", width=12)
l.grid (row=1, column=1, ipady=3)
l = Label(frameMenuCV2, text="Ret./Calc. Parent", width=12)
l.grid (row=2, column=1, ipady=3)
ButtonL = Button(frameMenuCV2, text="D.Pixels", height=1, width=7, foreground="white", background="blue")
ButtonL.bind("<Button>", showpix)
ButtonL.grid (row=1, column=2, ipadx=5, ipady=5)
ButtonM = Button(frameMenuCV2, text="D.Grains", height=1, width=7, foreground="black", background="grey")
ButtonM.bind("<Button>", showgraind)
ButtonM.grid (row=1, column=3, ipadx=5, ipady=5)
ButtonNN = Button(frameMenuCV2, text="P.Pixels", height=1, width=7, foreground="black", background="grey")
ButtonNN.bind("<Button>", showpixP)
ButtonNN.grid (row=2, column=2, ipadx=5, ipady=5)
ButtonN = Button(frameMenuCV2, text="P.Grains +", height=1, width=7, foreground="black", background="grey")
ButtonN.bind("<Button>", showgrainp)
ButtonN.grid (row=2, column=3, ipadx=5, ipady=5)
frameMenuCV2.grid(row=1, column=0)

listButtons1 = [ButtonA, ButtonB, ButtonC, ButtonD, ButtonE, ButtonF, ButtonG, ButtonDef, ButtonGOS]
listButtons2 = [ButtonL, ButtonM, ButtonNN, ButtonN]

frame3.grid (row=3, column=1, columnspan=4, ipadx=1, sticky=W)

frameRainbow = Frame(root)
LabelMin = Label(frameRainbow, textvariable=EMin, height=1)
LabelMin.grid (row=1, column=1, ipady=5)
imageRainbow = Image.new("RGB", (NbColors+1, 20), "white")
rainbow = ImageDraw.ImageDraw(imageRainbow)
for i in range(NbColors+1):
    rainbow.rectangle((i, 0, i + 1, 20), fill=Palette[i])
canvasRainbow = Canvas(frameRainbow, width=NbColors+1, height=20)
imageTkRainbow = ImageTk.PhotoImage(imageRainbow)
imageRainbowCV = canvasRainbow.create_image((NbColors+1) / 2, 10, image=imageTkRainbow)
canvasRainbow.grid (row=1, column=2)
LabelMax = Label(frameRainbow, textvariable=EMax, height=1)
LabelMax.grid (row=1, column=3, ipady=5)
frameRainbow.grid (row=4, column=1, columnspan=2)

frameCV = Frame(root)
root.bind("<Key>", presskey)
image = Image.new("RGB", (Nbpixy, Nbpixx), "white")
rainbow = ImageDraw.ImageDraw(image)
for i in range(NbColors+1):
    rainbow.rectangle((i * Nbpixx / 256, 0, (i + 1) * Nbpixx / 256, 10), fill=Palette[i])

canvas = Canvas(frameCV, width=Nbpixy, height=Nbpixx, scrollregion=(0, 0, 2 * Nbpixy, 2 * Nbpixx))

scry = Scrollbar(frameCV, orient=VERTICAL)
scry.grid(row=1, column=0, sticky="NS")
scry.config(command=canvas.yview)
canvas.config(yscrollcommand=scry.set)

scrx = Scrollbar(frameCV, orient=HORIZONTAL)
scrx.grid(row=0, column=1, sticky="EW")
scrx.config(command=canvas.xview)
canvas.config(xscrollcommand=scrx.set)

canvas.bind("<Motion>", motion)
canvas.bind("<Double-ButtonPress-1>", press)
canvas.bind("<ButtonPress-1>", selectzoneStart)
canvas.bind("<B1-Motion>", drawrectangle)
canvas.bind("<ButtonRelease-1>", selectzoneFinish)
canvas.bind("<Double-ButtonPress-3>", SaveCanvas)
canvas.grid (row=1, column=1)
frameCV.grid (row=5, column=1, sticky=W)

frameResults = Frame(root, borderwidth=2)
frameResults ["relief"] = RAISED
frameResults.grid (row=5, column=2, sticky=W)

frameMapInfo = Frame(frameResults, borderwidth=1, width=50)
frameMapInfo ["relief"] = SOLID
frameMapInfo.grid (row=0, column=0, sticky="W")

def OpenframeMapInfo(event):
    global frameMapInfo, InfoMap, POSX, POSY, E1, E2, E3, EX, EY, EZ, VariantNumber, QualInd, ButtonfMI
    try: ButtonfMI.destroy()
    except:pass
    l = Label(frameMapInfo, text="_____________ INFORMATION ___________", height=2)
    l.grid (row=0, column=1, columnspan=2, sticky="EW")
    ButtonfMI = Button(frameMapInfo, text="x", height=1, width=2)
    ButtonfMI["background"] = "grey"
    ButtonfMI.bind("<Button>", CloseframeMapInfo)
    ButtonfMI.grid (row=0, column=3, ipadx=1, sticky='W')
    InfoMap = StringVar()
    InfoMap.set("")
    LabelInfo = Label(frameMapInfo, textvariable=InfoMap, height=4, width=10)
    LabelInfo.grid (row=1, column=1, columnspan=3, sticky="EW")
    POSX = DoubleVar()
    POSX.set(str(0.))
    POSY = DoubleVar()
    POSY.set(str(0.))
    E1 = DoubleVar()
    E1.set(str(0.))
    E2 = DoubleVar()
    E2.set(str(0.))
    E3 = DoubleVar()
    E3.set(str(0.))
    EX = StringVar()
    EX.set(" [ ] ")
    EY = StringVar()
    EY.set(" [ ] ")
    EZ = StringVar()
    EZ.set(" [ ] ")
    VariantNumber = StringVar()
    VariantNumber.set("")
    QualInd = StringVar()
    QualInd.set("")
    LabelE1a = Label(frameMapInfo, text="Euler 1 = ", height=1)
    LabelE1a.grid (row=2, column=1, ipadx=5, sticky="W")
    LabelE1 = Label(frameMapInfo, textvariable=E1, height=1)
    LabelE1.grid (row=2, column=2, ipadx=5, sticky="W")
    LabelE2a = Label(frameMapInfo, text="Euler 2 = ", height=1)
    LabelE2a.grid (row=3, column=1, ipadx=5, sticky="W")
    LabelE2 = Label(frameMapInfo, textvariable=E2, height=1)
    LabelE2.grid (row=3, column=2, ipadx=5, sticky="W")
    LabelE3a = Label(frameMapInfo, text="Euler 3 = ", height=1)
    LabelE3a.grid (row=4, column=1, ipadx=5, sticky="W")
    LabelE3 = Label(frameMapInfo, textvariable=E3, height=1)
    LabelE3.grid (row=4, column=2, ipadx=5, sticky="W")
    LabelVariant = Label(frameMapInfo, text="Var. Number = ", height=1)
    LabelVariant.grid (row=6, column=1, ipadx=5, sticky="W")
    LabelVariant = Label(frameMapInfo, textvariable=VariantNumber, height=1)
    LabelVariant.grid (row=6, column=2, ipadx=5, sticky="W")
    LabelQual = Label(frameMapInfo, text="Quality = ", height=1)
    LabelQual.grid (row=7, column=1, ipadx=5, sticky="W")
    LabelQual = Label(frameMapInfo, textvariable=QualInd, height=1)
    LabelQual.grid (row=7, column=2, ipadx=5, sticky="W")

    LabelEXa = Label(frameMapInfo, text="X = ", height=1)
    LabelEXa.grid (row=8, column=1, ipadx=5, sticky="W")
    LabelEX = Label(frameMapInfo, textvariable=EX, height=1)
    LabelEX.grid (row=8, column=2, ipadx=5, sticky="W")
    LabelEYa = Label(frameMapInfo, text="Y = ", height=1)
    LabelEYa.grid (row=9, column=1, ipadx=5, sticky="W")
    LabelEY = Label(frameMapInfo, textvariable=EY, height=1)
    LabelEY.grid (row=9, column=2, ipadx=5, sticky="W")
    LabelEZa = Label(frameMapInfo, text="Z = ", height=1)
    LabelEZa.grid (row=10, column=1, ipadx=5, sticky="W")
    LabelEZ = Label(frameMapInfo, textvariable=EZ, height=1)
    LabelEZ.grid (row=10, column=2, ipadx=5, sticky="W")
    l = Label(frameMapInfo, text=" ", height=1)
    l.grid (row=11, column=8, columnspan=2, sticky="W")
    

def CloseframeMapInfo(event):
    global frameMapInfo, ButtonfMI
    frameMapInfo.destroy()
    frameMapInfo = Frame(frameResults, borderwidth=1, width=50)
    frameMapInfo ["relief"] = SOLID
    frameMapInfo.grid (row=0, column=0, sticky="W")
    ButtonfMI = Button(frameMapInfo, text="Pixel & Grain \n Info")
    ButtonfMI["background"] = "grey"
    ButtonfMI.bind("<Button>", OpenframeMapInfo)
    ButtonfMI.grid (row=0, column=3, ipadx=1, sticky='E')

OpenframeMapInfo(1)

frameGraphics = Frame(frameResults, borderwidth=2)
frameGraphics .grid (row=0, column=1, sticky="W")
frameHisto = Frame(frameGraphics , height=1, borderwidth=1) #,relief=SOLID
frameHisto.grid (row=0, column=0, columnspan=2, sticky="W")
frameStatistic = Frame(frameGraphics , height=1) #,relief=SOLID
frameStatistic.grid (row=1, column=0, columnspan=2, sticky="W")

#---------------------------------------------------------------------------

def BuildMenu():
    global bar

    #try: bar.destroy()
    #except: pass
    bar = Menu(root)
    filem = Menu(bar)
    filem.add_command(label="Import .ctf File", command=openFileHKL)
    filem.add_command(label="Import .ang File", command=openFileTSL)
##    filem.add_separator()
##    filem.add_command(label="Export HKL to TSL (.ctf => .ang)", command=HKLToTSL)
    filem.add_separator()
    filem.add_command(label="Open ARPGE File", command=openGrainFile)
    filem.add_command(label="Save as ARPGE File", command=SaveResults)
    filem.add_separator()
    filem.add_command(label="ReLoad ctf by changing Minor <-> Major Phases", command=ReLoadHKL)
##    filem.add_command(label="ReLoad ctf and keep only Daughter Phase", command=KeepOnlyDaughter)

    filterm = Menu(bar)
    filterm.add_command(label="CreateSubSet", command=create_SubSet0)
    filterm.add_separator()
    filterm.add_command(label="Remove Spikes", command=remspikes)
    filterm.add_command(label="Remove Tips", command=remtips)
    filterm.add_separator()
    filterm.add_command(label="Erode", command=erode)
    filterm.add_separator()
    filterm.add_command(label="Dilate Daughter Pixels but Keep Retained Parent", command=dilate0)
    filterm.add_command(label="Dilate Daughter Pixels but Keep Retained Parent, Iteratively", command=dilate)
    filterm.add_command(label="Dilate Daughter Pixels and Eat the Other Phases", command=dilate0_onAll)
    filterm.add_command(label="Dilate Daughter Pixels and Eat the Other Phases, Iteratively", command=dilate_onAll)
    filterm.add_separator()

    optionsm = Menu(bar)
    optionsm.add_command(label="Create/View with GenOVa", command=VariantProgram)
    optionsm.add_separator()
    optionsm.add_command(label="Burgers  bcc => hcp (Ti, Zr)", command=Burgers)
    optionsm.add_separator()
    optionsm.add_command(label="NW  fcc => bcc (Fe)", command=NW)
    optionsm.add_command(label="KS  fcc => bcc (Fe)", command=KS)
    optionsm.add_command(label="GT  fcc => bcc (Fe)", command=GT)
    optionsm.add_separator()
    optionsm.add_command(label="Zirconia  cubic => monoclinic", command=ZirconiaCM)
    optionsm.add_command(label="NiTi  B2 => B19'", command=NiTi)
    optionsm.add_separator()
    optionsm.add_command(label="Sigma3 exp(n) fcc => fcc => ...", command=Sigma3)
    optionsm.add_separator()
    optionsm.add_command(label="Others", command=LoadTransitionFiles)

    calculsf = Menu(bar)
    calculsf.add_command(label="Identify Grains", command=IdentifyDaughterGrains)
    calculsf.add_separator()
    calculsf.add_command(label="Ignore Small Grains", command=IgnoreSmallGrainsD)
    calculsf.add_separator()
    calculsf.add_command(label="Ignore Low Disorientations", command=IgnoreMisorientedGrainsD)
    calculsf.add_command(label="Ignore Low Disorientations Recursively", command=IgnoreMisorientedGrainsDIterate)
    calculsf.add_separator()
    calculsf.add_command(label="Calculate KAM (angles)", command=CalcMapDef)
    calculsf.add_command(label="Calculate KAM (angles + axes)", command=CalcMapDef2)
    calculsf.add_separator()
    if Transition != "Sigma3": calculsf.add_command(label="Draw Special Grain Boundaries", command=DrawGBOp)
    else:
        menuGBSigma3n = Menu(calculsf)
        menuGBSigma3n.add_command(label="n lower than 3", command=DrawGBSigma3n3)
        menuGBSigma3n.add_command(label="n lower than 4", command=DrawGBSigma3n4)
        menuGBSigma3n.add_command(label="n lower than 5", command=DrawGBSigma3n5)
        menuGBSigma3n.add_command(label="n lower than 6", command=DrawGBSigma3n6)
        calculsf.add_cascade(label="Draw Sigma3_exp(n) Grain Boundaries", menu=menuGBSigma3n)
    calculsf.add_command(label="Analyse Straight Grain Boundaries", command=ShowGBPlanes)
    calculsf.add_separator()

    calculsm = Menu(bar)
    if Transition == "Sigma3":
        calculsm.add_command(label="Reconstruct TRDs by Disreagarding Only Sigma3", command=ReconstTRD1)
        menuSigma3n = Menu(calculsm)
        menuSigma3n.add_command(label="n lower than 2", command=ReconstTRD2)
        menuSigma3n.add_command(label="n lower than 3", command=ReconstTRD3)
        menuSigma3n.add_command(label="n lower than 4", command=ReconstTRD4)
        calculsm.add_cascade(label="Reconstruct TRDs by Disreagarding Sigma3_exp(n) ", menu=menuSigma3n)
    else:
        calculsm.add_command(label="Reconstruct Parent Grains With Doublets", command=ReconstParentGrainsOrient2)
        calculsm.add_command(label="Reconstruct Parent Grains With Triplets", command=ReconstParentGrainsOrient3)
        calculsm.add_command(label="Reconstruct Parent Grains With Quadruplets", command=ReconstParentGrainsOrient4) 
    calculsm.add_separator()
    calculsm.add_command(label="Reset Reconstruction", command=ResetRec)
    if Transition == "Sigma3":
        calculsm.add_command(label="Calculate TRD Trees", command=TRDTrees)
    calculsm.add_separator()

    calculsp = Menu(bar)
    if Transition == "Sigma3":
        calculsp.add_command(label="Ignore Small TRDs", command=IgnoreSmallGrainsP)
        calculsp.add_separator()
        calculsp.add_command(label="Fractal. One TRD", command=ModeFractal)
        calculsp.add_command(label="Fractals. All TRDs", command=DrawFractalAll)
        calculsp.add_separator()
    else:
        calculsp.add_command(label="Ignore Small Grains", command=IgnoreSmallGrainsP)
        calculsp.add_separator()
        calculsp.add_command(label="Ignore Low Misorientations", command=IgnoreMisorientedGrainsP)
        calculsp.add_command(label="Ignore Low Misorientations Recursivily", command=IgnoreMisorientedGrainsPIterate)
        calculsp.add_command(label="Draw Sigma3 Misorientations in Red", command=DrawSigma3Aust)
        calculsp.add_command(label="Ignore Sigma3 Misorientations", command=IgnoreMisorientedGrainsPSigma3)
        calculsp.add_separator()
        calculsp.add_command(label="Polish Grain Boundaries", command=RemoveSpines)
        calculsp.add_command(label="Polish Grain Boundaries Recursively", command=RemoveSpinesRecurs)
        calculsp.add_separator()
        calculsp.add_command(label="Colorize White Holes", command=RemWhiteHoles)
        calculsp.add_command(label="Colorize White Holes Recursively", command=RemWhiteHolesRecurs)
        calculsp.add_separator()
        calculsp.add_command(label="Calculate Again Variant Indexes", command=AbsoluteVarNum)
        calculsp.add_command(label="Calculate Again Parent Orientations and Variant Indexes", command=RecalcOrientVarNum)
        calculsp.add_separator()
    if Transition == "Fe-NW" or Transition == "Fe-KS" or Transition == "Fe-GT" or Transition == "Fe-G" :
        calculsp.add_command(label="Show Low-Misoriention Bain's Packets", command=ShowBainZones0)
        calculsp.add_command(label="Show Close Packed Plane (CPP) Morito's Packets", command=ShowCPPPackets0)
        calculsp.add_command(label="Show Close Packed Direction (CPD) Packets", command=ShowCPDPackets0)
        calculsp.add_command(label="Show Plate (Burst) Packets", command=ShowBurstZone0)
        calculsp.add_separator()
    if Transition == "Fe-NW" or Transition == "Fe-KS"  or Transition == "Fe-GT" or Transition == "Fe-G" :
        calculsp.add_command(label="Show Pitsch - KS - NW in RGB mode", command=CalcTriangleOR)
        calculsp.add_separator()
    if Transition == "Zirconia-cubic-mono-type1" or Transition == "Zirconia-cubic-mono-type2" :
        calculsp.add_command(label="Show (100) Packets", command=ShowPackets100)
        calculsp.add_command(label="Show (010) Packets", command=ShowPackets010)
        calculsp.add_command(label="Show (001) Packets", command=ShowPackets010)
        calculsp.add_separator()
    calculsp.add_command(label="Calculate Parent Pixels", command=calculateTabPixP)
    calculsp.add_separator()
    calculsp.add_command(label="Export Parent Grains as text file (.ctf or .ang)", command=exportParentGrains)
    #calculsp.add_command(label="Export Parent Pixels as text file (.ctf or .ang)", command=exportParentPixels)
    calculsp.add_separator()
    if Transition == "Fe-NW" or Transition == "Fe-KS"  or Transition == "Fe-GT" or Transition == "Fe-G" :
        calculsp.add_command(label="Export Bain/CPP/CPD Packets as .ctf text file (packet number coded in the column BS)", command=ShowExportPackets)
        calculsp.add_separator()


    calculpf = Menu(bar)
    calculpf.add_command(label="All Daughter Pixels (slow)", command=DrawPoleFigDPix)
    calculpf.add_command(label="All Daughter Pixels (PF Density)", command=DrawPoleFigDPixDens)
    calculpf.add_separator()
    calculpf.add_command(label="All Daughter Grains", command=DrawPoleFigD)
    calculpf.add_command(label="All Daughter Grains (PF Density)", command=DrawPoleFigDDens)
    calculpf.add_separator()
    if Transition == "Sigma3": t1 = "Primary Grains"
    else: t1 = "Parent Grains"
    calculpf.add_command(label="All " + t1, command=DrawPoleFigP)
    calculpf.add_separator()
    if Transition == "Sigma3": t2 = "TRD"
    else: t2 = "Parent Grains"
    calculpf.add_command(label="Individual " + t2 + " (with its Daughter Grains)", command=ModePoleParentGrain)
    calculpf.add_command(label="Individual " + t2 + " (with its Pixels)", command=ModePoleParentGrain2)
    calculpf.add_separator()

    calculHP = Menu(bar)
    calculHP.add_command(label="Plot the Traces of the Habit Planes ", command=calculateHabitPlanes)
    if Transition == "NiTi": calculHP.add_command(label="Plot the Traces of the Junctions Planes ", command=calculateJunctionPlanes)
    calculHP.add_separator()
    

    statf = Menu(bar)
    statf.add_command(label="Daughter Grain Sizes", command=HistoGrainSizesD)
    if Transition == "Sigma3": t3 = "TRD Sizes"
    else: t3 = "Parent Grain Sizes"
    statf.add_command(label=t3, command=HistoGrainSizesP)
    statf.add_separator()
    statf.add_command(label="Disorientations between Neighb. Daughter Grains inside the Parent Grains", command=StatisticFreqMisD)
    #statf.add_command(label="Disorientations between All the Daughter Grains inside the Parent Grains", command=StatisticFreqMisDAll) # too long
    statf.add_separator()
    if Transition != "Sigma3":
        statf.add_command(label="Disorientations between Neighb. Parent Grains", command=StatisticFreqMisP)
        statf.add_command(label="Disorientations between All the Parent Grains", command=StatisticFreqMisPAll)
    if Transition == "Sigma3":
        statf.add_command(label="Sigma3n between Neighb. Daughter Grains (brute force)", command=StatisticSigma3nD)
    statf.add_separator()
    if Transition != "Sigma3":
        t4a = "Variants and Operators. One "
        t4b = "Operators. All "
    else:
        t4a = "Sigma3n Operators /Primary Grain. One "
        t4b = "Sigma3n and Operators /Primary Grains. All "
    statf.add_command(label=t4a + t2, command=ModeStatisticParentGrain)
    statf.add_command(label=t4b + t2, command=StatisticVariants)
    statf.add_separator()
    if Transition == "Sigma3":
            statf.add_command(label='Nb of Grains in TRDs', command=StatisticNbGinTRDs)
            statf.add_command(label='LLCs', command=StatisticLLC)
            statf.add_command(label='Twin Polysynthetism & Anistropy Parameters', command=StatisticsFractalAll) 
            statf.add_separator()

    param = Menu(bar)
    if Transition != "Sigma3":
        param.add_command(label="Load Parameters for Reconstruction", command=LoadParameters)
        param.add_command(label="Parameters for Automatic Reconstruction", command=AutoParamRec)
        param.add_command(label="Parameters for Grain Boundaries", command=AutoParamGB)
##        param.add_command(label="Parameters for Phases", command=AutoParamPhases)
        param.add_command(label="Parameters for Edition", command=AutoParamEdit)
        param.add_command(label="Parameters for Saving Images", command=AutoParamImage)
        param.add_command(label="Save Parameters for Reconstruction", command=SaveParameters)
        param.add_separator()
    else:
        param.add_command(label="Parameters for the automatic reconstruction of the TRDs", command=AutoParamSigma3n)
        param.add_command(label="Parameters for Saving Images", command=AutoParamImage)
        param.add_separator()

    infom = Menu(bar)
    infom.add_command(label="Help", command=User_Help)
    infom.add_separator()
    infom.add_command(label="Contact", command=User_Contact)
    infom.add_separator()
    infom.add_command(label="How to Refer to this Program", command=User_Refer)
    infom.add_separator()

    bar.add_cascade(label="Phase Transition", menu=optionsm)
    bar.add_cascade(label="Files", menu=filem)
    bar.add_cascade(label="Filters", menu=filterm)
    bar.add_cascade(label="Daughter Grains", menu=calculsf)
    if Transition == "Sigma3": bar.add_cascade(label="TRD Reconstruction", menu=calculsm)
    else: bar.add_cascade(label="Parent Reconstruction", menu=calculsm)
    if Transition == "Sigma3": bar.add_cascade(label="TRD", menu=calculsp)
    else: bar.add_cascade(label="Parent Grains", menu=calculsp)
    bar.add_cascade(label="Pole Figures", menu=calculpf)
    bar.add_cascade(label="Habit Planes",menu=calculHP)
    bar.add_cascade(label="Histograms", menu=statf)
    bar.add_cascade(label="Parameters", menu=param)
    bar.add_cascade(label="Information", menu=infom)

    root.config(menu=bar)

#---------------------------------------------------------------------------
BuildMenu()

root.mainloop()





