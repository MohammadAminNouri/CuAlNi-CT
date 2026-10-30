# C.Cayron, Sept.2008
# base to generate crystals: point groups, atomics positions
# remains: enter the atomic diffusion factors (cf Peng for example)
# 2019 prediction of the twin modes based on Friedel 1904
# 2020 weak planes and weak (=axial) twins

import os , string

from numpy import *
from numpy.linalg import inv, eig, det

from tkinter import *
from tkinter.filedialog import *
import tkinter.messagebox
import pickle
import copy
import time
from functools import reduce


m = array([[1, 0], [0, 1]])

from arpge.Algebra.CC_AlgebraicTools import rationalize,IntegriseAxe, ReduceAxe, cross, length, normalize, calculeRot, calculeMatRot, gcd3D, Bezout3D, LeftInv #= leftinverse = inverse for non-square matrix im.m = Id
from arpge.Utils.Files import getFullPath

try:
    directory = getFullPath("Crystallography")
    filename = 'PointGroups' + '.data'
    f = open(os.path.join(directory, filename), 'r')
    ListPG, DictPG, ListSyst, DictSyst, ListStruct, DictStruct, DictLatt = pickle.load(f)
    f.close()
except:
    import arpge.Crystallography.CC_PointGroups as CC_PointGroups
    print("LE PG CC_POINTGROUPS A ETE LANCE CAR LES FICHIERS DATA ETAIENT INEXISTANTS.")
    ListPG, DictPG, ListSyst, DictSyst, ListStruct, DictStruct, DictLatt = CC_PointGroups.build()

Degree = pi / 180.

Identity = array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

ListColor = ["blue", "red", "dark green", "brown", "yellow", "cyan", "violet"]



# ==============================================================================
# ==============================================================================
# attention il existe dans SciPy une pg qui sappelle crystal.py et il met le bordel
# j ai du tout renommer tous mes objets en Crystal() et non crystal() pour cette raison

class Atom:
    #dot dans les cliche de diffraction
    def __init__(self, el, x, y, z, r=1, col="brown", wick=1):
        #wick for wickoff, 1 si l atome genere les autres, 0 pour ceux qui ont ete generes
        #grainp = parent grain
        self.el = el
        self.x = x
        self.y = y
        self.z = z
        self.r = r
        self.col = col
        self.wick = wick
    def toList(self):
        el = self.el
        x = self.x
        y = self.y
        z = self.z
        r = self.r
        col = self.col
        wick = self.wick
        return [el, x, y, z, r, col, wick]

def fdiff(el, theta=0):
    fdiff = 1
    # a completer
    # a ameliorer avec Weikenmeir-Kholl
    if el == "H": fdiff = 1
    elif el == "X": fdiff = 1
    elif el == "Li": fdiff = 3
    elif el == "Be": fdiff = 4
    elif el == "B": fdiff = 5
    elif el == "C":fdiff = 6
    elif el == "N": fdiff = 7
    elif el == "O": fdiff = 8
    elif el == "F": fdiff = 9
    elif el == "Ne": fdiff = 10
    elif el == "Na": fdiff = 11
    elif el == "Mg": fdiff = 12
    elif el == "Al":fdiff = 13
    elif el == "Si":fdiff = 14
    elif el == "P": fdiff = 15
    elif el == "S": fdiff = 16
    elif el == "Cl": fdiff = 17
    elif el == "Ar": fdiff = 18
    elif el == "K": fdiff = 19
    elif el == "Ca": fdiff = 20
    elif el == "Sc": fdiff = 21
    elif el == "Ti": fdiff = 22
    elif el == "V": fdiff = 23
    elif el == "Cr": fdiff = 24
    elif el == "Mn": fdiff = 25
    elif el == "Fe": fdiff = 26
    elif el == "Co": fdiff = 27
    elif el == "Ni":fdiff = 28
    elif el == "Cu": fdiff = 29
    elif el == "Zn": fdiff = 30
    elif el == "Ga": fdiff = 31
    elif el == "Ge": fdiff = 32
    elif el == "As": fdiff = 33
    elif el == "Se": fdiff = 34
    elif el == "Br": fdiff = 35
    elif el == "Kr": fdiff = 36
    elif el == "Rb": fdiff = 37
    elif el == "Sr": fdiff = 38
    elif el == "Y":fdiff = 39
    elif el == "Zr":fdiff = 40
    elif el == "Nb": fdiff = 41
    elif el == "Mo": fdiff = 42
    elif el == "Tc": fdiff = 43
    elif el == "Ru": fdiff = 44
    elif el == "Rh":fdiff = 45
    elif el == "Pd":fdiff = 46
    elif el == "Ag":fdiff = 47
    elif el == "Cd":fdiff = 48
    elif el == "In":fdiff = 49
    elif el == "Sn":fdiff = 50
    elif el == "Sb":fdiff = 51
    elif el == "Te":fdiff = 52
    elif el == "I":fdiff = 53
    elif el == "Xe":fdiff = 54
    elif el == "Cs":fdiff = 55
    elif el == "Ba":fdiff = 56
    elif el == "La":fdiff = 57
    elif el == "Gd":fdiff = 64
    elif el == "Yb":fdiff = 70
    elif el == "Ta":fdiff = 73
    elif el == "W":fdiff = 74
    elif el == "Ir":fdiff = 77
    elif el == "Pt":fdiff =78
    elif el == "Au":fdiff = 79
    elif el == "Pb":fdiff = 82
    else: print("this atom was not entered in the table")
    
    return fdiff

def transAtom(atom, v):
    el, x, y, z, r, col, wick = atom.toList()
    newatom = Atom(el, x + v[0], y + v[1], z + v[2], r, col, wick)
    return newatom

def rotAtom(atom, Rot):
    el, x, y, z, r, col, wick = atom.toList()
    xr, yr, zr = dot(Rot, array([x, y, z]))
    newatom = Atom(el, xr, yr, zr, r, col, wick)
    return newatom

# ==============================================================================

class Crystal:

    directory = getFullPath(os.path.join("Crystallography", "PhasesCRYST"))

    def __init__(self, ELEMENT="Cu", A=3.6, B=3.6, C=3.6, ALPHA=90 * Degree, BETA=90 * Degree,
                 GAMMA=90 * Degree, SYST="Cubic", STRUCT="FCC", PG = '1', STIFF=array(identity(6)), listAtoms=[]):
        self.el = ELEMENT
        self.a, self.b, self.c = A, B, C
        self.alpha, self.beta, self.gamma = ALPHA, BETA, GAMMA
        self.syst = SYST
        self.struct = STRUCT
        self.pg = PG
        self.stiff = STIFF
        self.listAtoms = listAtoms


    def __call__(self, ELEMENT="Cu", A=3.6, B=3.6, C=3.6, ALPHA=90 * Degree, BETA=90 * Degree,
                 GAMMA=90 * Degree, SYST="Cubic", STRUCT="FCC", PG ='1', STIFF=array(identity(6)), listAtoms=[]):
        self.el = ELEMENT
        self.a, self.b, self.c = A, B, C
        self.alpha, self.beta, self.gamma = ALPHA, BETA, GAMMA
        self.syst = SYST
        self.struct = STRUCT
        self.pg = PG
        self.stiff = STIFF
        self.listAtoms = listAtoms

    def sym(self): #liste des matrices du groupe ponctuel (dans l espace direct)
        return DictPG[self.pg]

    def symrec(self):  #liste des matrices du groupe ponctuel dans l espace reciproque INUTILE = sym()
        listsymrec = []
        for sdir in self.sym():
            srec = transpose(inv(sdir))
            srec = array([[int(round(x)) for x in l] for l in srec])
            listsymrec.append(srec)
        return listsymrec

    def compliance(self):
        return inv(self.stiff)

    def toList(self):
        el = self.el
        a, b, c = self.a, self.b, self.c
        alpha, beta, gamma = self.alpha, self.beta, self.gamma
        syst = self.syst
        struct = self.struct
        PG = self.pg
        stiff = self.stiff
        listAtoms = self.listAtoms
        return el, a, b, c, alpha, beta, gamma, syst, struct, PG, stiff, [at.toList() for at in listAtoms]

    def exportCryst(self, directory):
        listcryst = self.toList()
        crystname = self.el + '.cryst'
        os.chdir(directory)
        r = True
        if crystname in os.listdir(directory):
            r = tkinter.messagebox.askyesno('Warning', 'Overwrite the existing file' + crystname + ' ?')
        if r:
            f = open(crystname, 'w')
            pickle.dump(listcryst, f)
            f.close()

    def metricTensDir(self):  # metric tensor :           REC -> DIR
        a, b, c = self.a, self.b, self.c
        alpha, beta, gamma = self.alpha, self.beta, self.gamma
        metricTensDir = array([[a ** 2, a * b * cos(gamma), a * c * cos(beta)],
                               [a * b * cos(gamma), b ** 2, b * c * cos(alpha)],
                               [a * c * cos(beta), b * c * cos(alpha), c ** 2]])
        return metricTensDir

    def structTensDir(self): # structure tensor/DIR :    ORTHO -> DIR
        a, b, c = self.a, self.b, self.c
        alpha, beta, gamma = self.alpha, self.beta, self.gamma
        dst = sqrt(1 + 2 * cos(alpha) * cos(beta) * cos(gamma) - cos(alpha) ** 2 - cos(beta) ** 2 - cos(gamma) ** 2)
        structTensDir = array([[a * sin(beta), b * (cos(gamma) - cos(alpha) * cos(beta)) / sin(beta), 0],
                               [0, b * dst / sin(beta), 0],
                               [a * cos(beta), b * cos(alpha), c]])
            
        return structTensDir

    def metricTensRec(self):# inv metric tensor :       DIR -> REC 
        metricTensRec = inv(self.metricTensDir)
        return metricTensRec

    def structTensRec(self):# structure tensor/REC:     ORTHO -> REC
        structTensRec = inv(transpose(self.structTensDir))
        return structTensRec

    def Fhkl(self, h, k, l):
        Fhkl = 0
        for at in self.listAtoms:
            el = at.el
            x, y, z = at.x, at.y, at.z
            Fhkl = Fhkl + fdiff(el) * exp(2 * math.pi * 1j * (h * x + k * y + l * z))
        return Fhkl

    def holopg(self): # matrices of the holoedric point group 
        if self.syst=="Triclinic": h = "-1"
        elif self.syst=="Monoclinic": h = "2/m"
        elif self.syst=="Orthorhombic": h = "mmm"
        elif self.syst=="Tetragonal": h = "4/mmm"
        elif self.syst=="Trigonal": h = "-3m"
        elif self.syst=="Hexagonal": h = "6/mmm"
        elif self.syst=="Cubic": h = "m3m"
        return h
    
    def holosym(self):
        return DictPG[self.holopg()]

    def holosymrec(self):
        listsymrec = []
        for sdir in self.holosym():
            srec = transpose(inv(sdir))
            srec = array([[int(round(x)) for x in l] for l in srec])
            listsymrec.append(srec)
        return listsymrec


    def listduvw(self, remsym=1, distmax=20):
        indmax = [0, 0, 0]
        for n in range(3):
            flag = 1
            i = 1
            while flag:   # calcul de l indice max sur les u v w
                if n == 0: uvw = [i, 0, 0]
                elif n == 1: uvw = [0, i, 0]
                elif n == 2: uvw = [0, 0, i]
                guvw = vectDir(self, uvw)
                u, v, w = uvw
                duvw = guvw.length()
                if duvw > distmax:
                    flag = 0
                    indmax[n] = i
                i = i + 1
                if i>20:break
        print(indmax)
        listguvwCompl = []
        if self.syst in ['Triclinic', 'Monoclinic'] or remsym == 0:
            ru = list(range(-indmax[0], indmax[0]))
            rv = list(range(-indmax[1], indmax[1]))
            rw = list(range(-indmax[2], indmax[2]))
            ru.sort(reverse=True)
            rv.sort(reverse=True)
            rw.sort(reverse=True)
        else :
            ru = list(range(indmax[0]))
            rv = list(range(indmax[1]))
            rw = list(range(indmax[2]))
        for u in ru:
            for v  in rv:
                for w in rw:
                    listguvwCompl.append([u, v, w])
        try: listguvwCompl.remove([0, 0, 0])
        except: pass
    
        if remsym == 0:
            listguvw = listguvwCompl
        else: # facon crible d Eratostene
            listsym = self.sym()
            listguvwProv = copy.deepcopy(listguvwCompl)
            listguvw = []
            while listguvwProv:
                guvw = listguvwProv[0]
                listguvw.append(guvw)
                for s in listsym:
                    newguvw = dot(s, array(guvw)).tolist()
                    try: listguvwProv.remove(newguvw)
                    except:pass
        listuvw = []
        for uvw in listguvw:
            guvw = array(uvw)
            u, v, w = uvw
            #invdhkl =  ghkl.length()
            duvw = sqrt (dot (guvw, dot (self.metricTensDir, guvw)))
            if duvw < distmax: listuvw.append([duvw, u, v, w, 0])
        listuvw.sort(reverse=False)

        return listuvw   

    def listdhkl(self, remsym=1, freqmax=1):
        indmax = [0, 0, 0]
        for n in range(3):
            flag = 1
            i = 1
            while flag:   # calcul de l indice max sur les h k l
                if n == 0: hkl = [i, 0, 0]
                elif n == 1: hkl = [0, i, 0]
                elif n == 2: hkl = [0, 0, i]
                ghkl = vectRec(self, hkl)
                h, k, l = hkl
                invdhkl = ghkl.length()
                if invdhkl > freqmax:
                    flag = 0
                    indmax[n] = i
                i = i + 1
        listghklCompl = []
        if self.syst in ['Triclinic', 'Monoclinic'] or remsym == 0:
            rk = list(range(-indmax[0] + 1, indmax[0]))
            rh = list(range(-indmax[1] + 1, indmax[1]))
            rl = list(range(-indmax[2] + 1, indmax[2]))
            rk.sort(reverse=True)
            rh.sort(reverse=True)
            rl.sort(reverse=True)
        else :
            rk = list(range(indmax[0]))
            rh = list(range(indmax[1]))
            rl = list(range(indmax[2]))
        for k in rk:
            for h  in rh:
                for l in rl:
                    listghklCompl.append([h, k, l])
        listghklCompl.remove([0, 0, 0])
        if remsym == 0:
            listghkl = listghklCompl
        else: # facon crible d Eratostene
            listsymrec = self.symrec()
            listghklProv = copy.deepcopy(listghklCompl)
            listghkl = []
            while listghklProv:
                ghkl = listghklProv[0]
                listghkl.append(ghkl)
                for s in listsymrec:
                    newghkl = dot(s, array(ghkl)).tolist()
                    try: listghklProv.remove(newghkl)
                    except:pass
        listhkl = []
        for hkl in listghkl:
            ghkl = array(hkl)
            h, k, l = hkl
            #invdhkl =  ghkl.length()
            invdhkl = sqrt (dot (ghkl, dot (self.metricTensRec, ghkl)))
            if invdhkl < freqmax: listhkl.append([1. / invdhkl, h, k, l, abs(self.Fhkl(h, k, l))**2])
        listhkl.sort(reverse=True)
        return listhkl

    def listangles(self, freqmax=1):
        indmax = [0, 0, 0]
        for n in range(3):
            flag = 1
            i = 1
            while flag:   # calcul de l indice max sur les h k l
                if n == 0: hkl = [i, 0, 0]
                elif n == 1: hkl = [0, i, 0]
                elif n == 2: hkl = [0, 0, i]
                ghkl = vectRec(self, hkl)
                invdhkl = ghkl.length()
                if invdhkl > freqmax:
                    flag = 0
                    indmax[n] = i
                i = i + 1
        M = 3
        indmax = [min(x, M) for x in indmax]
        listsymrec = self.symrec()
        print(len(listsymrec))
        listghkl = []
        if 1:
            rh = list(range(-indmax[0] + 1, indmax[0]))
            rk = list(range(0, indmax[1]))
            rl = list(range(-indmax[2] + 1, indmax[2]))
            rh.sort(reverse=True)
            rk.sort(reverse=True)
            rl.sort(reverse=True)
        for h in rh:
            for k  in rk:
                for l in rl:
                    listghkl.append([h, k, l])
        while [0, 0, 0] in listghkl: listghkl.remove([0, 0, 0])
        for hkl in listghkl:
            h, k, l = hkl
            if [-h, -k, -l] in listghkl: listghkl.remove([-h, -k, -l])
        listgcouple = []
        for i in range(len(listghkl)):
            hkl1 = listghkl[i]
            for j in range(i):
                hkl2 = listghkl[j]
                listgcouple.append([hkl1, hkl2])

        listangles = []
        listgcoupleProv = copy.deepcopy(listgcouple)

        while listgcoupleProv:
            #print len(listgcoupleProv)
            hkl1, hkl2 = listgcoupleProv[0]
            a = angle(hkl1, hkl2, self.metricTensRec)
            if a > 0.0001: listangles.append([a, hkl1, hkl2])
            for s in listsymrec:
                newhkl1 = dot(s, array(hkl1)).tolist()
                newhkl2 = dot(s, array(hkl2)).tolist()
                try: listgcoupleProv.remove([hkl1, hkl2])
                except:pass

        listangles.sort()
        return listangles

    compl = property(compliance, doc="compliance calculated from stiffness")
    metricTensDir = property(metricTensDir, doc="metric tensor calculated from the crystal parameters")
    structTensDir = property(structTensDir, doc="structural tensor calculated from the crystal parameters")
    metricTensRec = property(metricTensRec, doc="metric tensor calculated from inv of metric tensor")
    structTensRec = property(structTensRec, doc="structural tensor calculated from inv of structural tensor")

    def _str_(self):
        return (self.a).tostring()
    def _repr_(self):
        return (self.a).tostring()

# ==============================================================================
def my_allclose(a,b,err=0.00001): # a ,b matrices, allclose does not work well when the matrices contains -0 and +0
    flag = 1
    af, bf = a.flatten().tolist(), b.flatten().tolist()
    for i in range(len(af)):
        if abs(af[i]-bf[i])>err:
            flag = 0
            break
    return flag
    
# ==============================================================================

def Inter(q1,q2):
    # regarde si l'interection de deux list contient au moins un element
    
    flag=0
    for m1 in q1:
        for m2 in q2:
            if allclose(m1,m2):
                flag=1
                break
    return flag

# ==============================================================================

def IsInGroup(m,l):
    #indice de la matrice m dans la list l
    res = 0
    for li in l:
            if allclose(m,li):res=1
            break
    return res

# ==============================================================================

def cut(nb, n=6):
    if int(nb)==nb: return str(nb)
    else:
        t = str(round(nb,8))
        if "e-" in t: t = "0."
        else: t = t[:t.find(".") + n]
        return t

# ==============================================================================
def tabify(s, tabsize = 15):
    ln = ((len(s)/tabsize)+1)*tabsize
    return s.ljust(ln)   

# ==============================================================================

def ft_min(x):
    if int(x)==x: return x,0
    else:
        n = int(round(x))
        r = round(x-n,10)
        
        return n,r
       
# ==============================================================================

def fromList(listcryst):
    # return a crystal object from the list of its parameters
    lc = listcryst
    element, a, b, c, alpha, beta, gamma, syst, struct, PG, stiff = lc[0], lc[1], lc[2], lc[3], lc[4], lc[5], lc[6], lc[7], lc[8], lc[9], lc[10]
    listAtoms = []
    for at in lc[11]:
        listAtoms.append(Atom(*at))
    cryst = Crystal(element, a, b, c, alpha, beta, gamma, syst, struct, PG, stiff, listAtoms)
    return cryst

# ==============================================================================    

def importCryst(el, directory):
    # el = name of the crystal to be loaded
    # directory = directory of the el.cryst file

    os.chdir(directory)
    f = open(el + '.cryst', 'r')
    lc = pickle.load(f)
    cryst = fromList(lc)
    f.close()

    return cryst

# ==============================================================================

class cloneCryst(Crystal):
    def __init__(self, cryst):
        self.el = cryst.el
        self.a, self.b, self.c = cryst.a, cryst.b, cryst.c
        self.alpha, self.beta, self.gamma = cryst.alpha, cryst.beta, cryst.gamma
        self.syst = cryst.syst
        self.struct = cryst.struct
        self.pg = cryst.pg
        self.stiff = cryst.stiff
        self.listAtoms = cryst.listAtoms

# ==============================================================================

class vectDir:
    def __init__(self, cryst, U):
        global CrystR
        self.u, self.v, self.w = U[0], U[1], U[2]
        self.cryst = cryst
        CrystR = cryst
    def toList(self):
        u, v, w = self.u, self.v, self.w
        return [u, v, w]
    def toArray(self):
        u, v, w = self.u, self.v, self.w
        return array([u, v, w])
    def length(self):
        u, v, w = self.u, self.v, self.w
        vect = array([u, v, w])
        return sqrt (dot (vect, dot (CrystR.metricTensDir, vect)))
    def normalize(self):
        u, v, w = self.u, self.v, self.w
        vect = array([u, v, w])
        le = self.length()
        if le != 0: vect = divide (vect, le)
        else: raise ZeroDivisionError("Can't normalize a zero-length vector")
        return vect
    def toRec(self): #calculate the coordonates in the reciprocal space 
        u, v, w = self.u, self.v, self.w
        vect = array([u, v, w])
        vrec = dot(CrystR.metricTensDir, vect)
        vrec = vectRec(CrystR, vrec)
        return vrec
    def toOrtho(self): #calculate the coordonates in the "structure" orthonormal space
        u, v, w = self.u, self.v, self.w
        vect = array([u, v, w])
        vortho = dot(CrystR.structTensDir, vect)
        return vortho

# ==============================================================================

class vectRec:
    def __init__(self, cryst, H):
        global CrystR
        self.h, self.k, self.l = H[0], H[1], H[2]
        self.cryst = cryst
        CrystR = cryst
    def toList(self):
        h, k, l = self.h, self.k, self.l
        return [h, k, l]
    def toArray(self):
        h, k, l = self.h, self.k, self.l
        return array([h, k, l])
    def length(self):
        h, k, l = self.h, self.k, self.l
        vect = array([h, k, l])
        return sqrt (dot (vect, dot (CrystR.metricTensRec, vect)))
    def normalize(self):
        h, k, l = self.h, self.k, self.l
        vect = array([h, k, l])
        le = self.length()
        if le != 0: vect = divide (vect, le)
        else: raise ZeroDivisionError("Can't normalize a zero-length vector")
        return vect
    def toDir(self): #calculate the coordonates in the direct space 
        h, k, l = self.h, self.k, self.l
        vect = array([h, k, l])
        vdir = dot(CrystR.metricTensRec, vect)
        vdir = vectDir(CrystR, vdir)
        return vdir
    def toOrtho(self): #calculate the coordonates in the "structure" orthonormal space
        h, k, l = self.h, self.k, self.l
        vect = array([h, k, l])
        vortho = dot(CrystR.structTensRec, vect)
        return vortho

# ==============================================================================

def listdir(p,n,umax):   # not used
    # list of direction that belongs to a plane p=(h,k,l) with an index n: uh+vk+wl = n such that u,v,w < umax
    h,k,l = p
    listd = []
    for u in range(-umax,umax+1):
        for v in range(-umax,umax+1):
            for w in range(-umax,umax+1):
                if u*h+v*k+w*l == n: listd.append([u,v,w])
    return listd
   
# ==============================================================================

def angleVect(cryst, vect1, vect2):
    # return an angle between two direction in degree

    ang = 0
    flag = False

    try:
        if not(str(vect1.cryst.el()) == str(vect2.cryst.el()) == str(cryst.el())):
            tkinter.messagebox.showinfo("problem", "the crystal for each direction is not the same")
    except: pass

    try:
        if vect1.__class__.__name__ == "vectDir":
            try:
                if vect2.__class__.__name__ == "vectDir":
                    vect2 = vect2.toRec()
                elif vect2.__class__.__name__ == "list":
                    vect2 = vectRec(cryst, vect2)
            except:
                # a priori cest donc un array
                if "array" in str(type(vect2)): vect2 = vectRec(cryst, vect2.tolist())
                else : print("----------erreur-----", type(vect2))
            flag = True

        elif vect1.__class__.__name__ == "vectRec":
            try:
                if vect2.__class__.__name__ == "vectRec":
                    vect2 = vect2.toDir()
                elif vect2.__class__.__name__ == "list":
                    vect2 = vectDir(cryst, vect2)
            except:
                if "array" in str(type(vect2)): vect2 = vectDir(cryst, vect2.tolist())
                else : print("----------erreur-----", type(vect2))
            flag = True

        elif vect1.__class__.__name__ == "list" or "array" in str(type(vect1)):
            vect1 = vectDir(cryst, vect1)
            try:
                if vect2.__class__.__name__ == "vectDir":
                    vect2 = vect2.toRec()
                elif vect2.__class__.__name__ == "list":
                    vect2 = vectRec(cryst, vect2)
            except:
                # a priori cest donc un array
                if "array" in str(type(vect2)): vect2 = vectRec(cryst, vect2.tolist())
                else : print("----------erreur anglevect-----", type(vect2))
            flag = True

    except:
        if "array" in str(type(vect1)): vect1 = vectDir(cryst, vect1.tolist())
        if "array" in str(type(vect2)): vect2 = vectDir(cryst, vect2.tolist())
        ang = angleVect(cryst, vect1, vect2)

    if flag:
        try: ang = arccos(dot(vect1.toArray(), vect2.toArray()) / (vect1.length()*vect2.length())) / Degree
        except : ang = 0.
        if isnan(ang): ang = 0
        # en effet il se peut que le produit soit tres legerement superieur a 1 si les deux vecteurs sont
        # egaux, donc je mets leur angle directement a 0
    return ang


# ==============================================================================
#   ET QUELQUES PROCEDURES QUI NE SONT PAS SOUS FORME DE CLASSES

# ==============================================================================

def lengthCryst (v, metricTensor):

# Calculate the length of a vector expressed in Crystal coordinates.

    v = array (v)

    return sqrt (dot (v, dot (metricTensor, v)))


# ==============================================================================

def normalizeCryst (v, metricTensor):

# Normalize a vector expressed in Crystal coordinates.

    v = array (v)
    length = lengthCryst (v, metricTensor)
    if length != 0: v = divide (v, length)
    else: raise ZeroDivisionError("Can't normalize a zero-length vector")

    return v


### ==============================================================================

def cosinus (axe1, axe2, metricTensor):

    d1 = lengthCryst (axe1, metricTensor)
    d2 = lengthCryst (axe2, metricTensor)
    cos = dot (axe2, dot(metricTensor, axe1)) / (d1 * d2)
    if cos > 1: cos = 1
    if cos < -1: cos = -1

    return cos

 # ==============================================================================

def angle (axe1, axe2, metricTensor=Identity, axeRef=[0, 0, 0],mode = "deg"):

    cos	 = cosinus (axe1, axe2, metricTensor)
    angle	 = arccos (cos)
    if axeRef != [0, 0, 0]: dett = det(array([axe1, axe2, axeRef]))
    else: dett = det(array([axe1, axe2, [1, 0, 0]]))
    if dett < 0: angle = -angle
    elif dett == 0:
        AxeRef2 = [0, 1, 0]
        dett = det(array ([axe1, axe2, [0, 1, 0]]))
        if dett < 0: angle = -angle
        elif dett == 0:
                dett = det(array ([axe1, axe2, [0, 0, 1]]))
                if dett < 0: angle = -angle
    if abs(angle + math.pi) < 0.0001: angle = math.pi
    if mode=="deg": angle = angle * 180. / math.pi
    return angle

# ==============================================================================

def angle_2 (axe1, axe2, metricTensor, axeRef=[0, 0, 0]):

    cos = cosinus (axe1, axe2, metricTensor)
    ang = arccos (cos)
    if abs(ang) > 0.000001:
        dett = det(array ([axe1, axe2, cross(axe1, axe2)]))
        if dett < 0: ang = -ang
    if abs(ang + math.pi) < 0.0001: ang = math.pi
    ang = ang / Degree
    return ang

#====================================================================================================   

def SimpleAxe(axe, GD):
    # GD = groupe of rotations
    nbsignesref = 0
    laxe = []
    axeres = axe
    if len(axe)>0:
        for sd in GD:
            naxe = dot(sd, array(axe))
            naxe = naxe.tolist()
            if naxe not in laxe: laxe.append(naxe)
        for axe in laxe:# affiche l angle le plus petit mais l axe qui a le maximum de signes positifs
            nbsignes = add.reduce([int(x == abs(x)) for x in axe]) #compte les signes +
            if nbsignes > nbsignesref:
                axeres = axe
                angleres = angle
                nbsignesref = nbsignes
        return axeres
    else:
        return []

#==============================================================================

def SimpleRotMin(lrot, GPrec):
    u = array([0, 0, 1])
    v = array([1, 1, 1])
    lrotsimpl = []
    lrotang = []
    angmin = 180
    axemin = [0, 0, 1]
    for couple in lrot:
        ang = couple[0]
        if ang < angmin - 0.00001:
            angmin = ang
            axemin = couple[1]
        elif abs(angmin - ang) < 0.00001:
            axe = couple[1]
            p1 = abs(angle(axe, u, GPrec))
            p2 = abs(angle(axemin, u, GPrec))
            if p1 < p2 - 0.00001:
                angmin = ang
                axemin = axe
            elif abs(p2 - p1) < 0.00001:
                p1p = abs(angle(axe, v, GPrec))
                p2p = abs(angle(axemin, v, GPrec))
                if p1p < p2p:
                    angmin = ang
                    axemin = axe

    return [angmin, axemin]

#===========================================================================================

def Disorientation(m, listsym):
    angMin = 361
    dism = []
    symm = []
    for symk in listsym: # voir Heinz et Neumann pour les classes d angles
        mk = dot(m,symk)
        tr = trace(mk)
        if tr < -1: tr = -1
        elif tr > 3: tr = 3
        arand = abs(arccos((tr - 1) / 2)) * 180. / pi
        if arand < angMin:
            angMin = arand
            dism = mk
            symm = symk

    return dism

#===========================================================================================
def getKey(item):
	return item[0]
    
#===========================================================================================

def ToFourIndexDir(axis):
    u,v,w = axis
    nu = (2*u-v)
    nv = (2*v-u)
    t = -(nu+nv)
    nw = 3*w
    return array([nu,nv,t,nw])

#===========================================================================================

def ToThreeIndexDir(axis):
    u,v,t,w = axis
    nu = (2*u+v)
    nv = (2*v+u)
    return array([nu,nv,w])

#===========================================================================================

def ToFourIndexPlane(p):
    h,k,l = p
    t = -(h+k)
    return array([h,k,t,l])

#===========================================================================================

def ToThreeIndexPlane(p):
    h,k,t,l = p
    return array([h,k,l])

#===========================================================================================

def ListEquivRot(m, listsym,TS,savemk=0,allaxes=0): # m = matrix DIR to DIR
    
    def IsEquiv(ang0,axis0):
        r=0
        for l in listrot:
            ang,axe = l[0],l[1]
            if abs(ang0-ang)<0.000001:
                for symk in listsym:
                    eqaxis= dot(symk,axis0)
                    if length(eqaxis-axe)<0.000001:
                        r=1
                        break
            if r: break
        return r
            
    listrot = []
    for symk in listsym: # voir Heinz et Neumann pour les classes d angles
        mk = dot(symk,m)
        mkortho = dot(TS,dot(mk,inv(TS))) # to transform it into ORTHO to ORTHO
        rot = calculeRot(mkortho)
        if rot:
            if rot[1]:
                ang,axe= [rot[0],IntegriseAxe(dot(inv(TS),rot[1]),0.05)]
                if allaxes==1 or not IsEquiv(ang,axe):
                    if not savemk: listrot.append([ang,axe,mkortho])
                    else: listrot.append([ang,axe,mk])
    try: listrot.sort()
    except: pass
    #sorted(listrot,key=getKey)
    return listrot

#===========================================================================================

def ShearBevisCrocker(C,cryst): # C = correspondence matrix
    res2 = trace(dot(dot(cryst.metricTensRec ,transpose(C)),dot(cryst.metricTensDir,C)))-3
    if res2>=0: res = sqrt(res2)
    else: res = 0
    return res

def ShearMyFormula(F,cryst): # F = distortion matrix
    return sqrt(trace(dot(dot(cryst.metricTensDir ,(F-Identity)),dot(cryst.metricTensRec,transpose(F-Identity)))))

def ShearMyFormulaB(F,cryst): # F = distortion matrix
    return sqrt(trace(dot(transpose((F-Identity)),dot(cryst.metricTensDir, (F-Identity)))))

#===========================================================================================                        

def crystWindow(CRYST0,root,top=1):
    global CRYST, frame
    global fpar, fcryst, list_crystsyst, list_crystpg, list_cryststruct
    global syst, scrollpg, scrollstruct, freqmax, distmax, fcalc

    if top:
        frame = Toplevel(root)
    else:
        frame = Frame(root) #Toplevel(root)
        frame.grid(row=0, column=1, columnspan=2, sticky='W')
        
    frame["relief"] = RAISED

    Directory = getFullPath("Crystallography")
    CRYST = CRYST0

    tkSym = IntVar()

    colorg = "grey90"
    freqmax = 1. / (1.5)
    distmax = 10.

    def open_cryst():
        global fpar, CRYST, fcalc
        os.chdir(os.path.join(Directory, "phasesCRYST"))
        crystname = askopenfilename(filetypes=[("crystals", "*.cryst"), ("All files", "*.*")])
        while ("/" in crystname) :
            ref = string.find(crystname, "/")
            crystname = crystname[ref + 1:]
        crystname = crystname[:string.find(crystname, ".")]
        os.chdir(Directory)
        print("crystname", crystname)
        try: fcalc.destroy()
        except:pass

        if crystname:
            try:
                CRYST = importCryst(crystname, os.path.join(Directory, "phasesCRYST"))
                print("***", CRYST.el)
            except :
                print("MODIFY THE CRYSTAL")
                CRYST = importCryst("Cu", os.path.join(Directory, "phasesCRYST"))
                print("---", CRYST.el)
        fpar.destroy()
        buildFrame()

    def calc_convert():
        global fcalc, uvw, hkl, entry_uvw, entry_hkl
        global uvw4, hkl4, entry_uvw4, entry_hkl4

        try: fcalc.destroy()
        except:pass

        uvw = [0, 0, 0]
        hkl = [0, 0, 0]
        uvw4 = [0, 0, 0, 0]
        hkl4 = [0, 0, 0, 0]

        def attr_calc(event):
            global uvw, hkl,uvw4,hkl4
            uvw = []
            for i in range(3): uvw.append(string.atoi(entry_uvw[i].get()))
            uvw = array(uvw)
            hkl = []
            for i in range(3): hkl.append(string.atoi(entry_hkl[i].get()))
            hkl = array(hkl)
            uvw4 = []
            for i in range(4): uvw4.append(string.atoi(entry_uvw4[i].get()))
            uvw4 = array(uvw4)
            hkl4 = []
            for i in range(4): hkl4.append(string.atoi(entry_hkl4[i].get()))
            hkl4 = array(hkl4)

        def calcuvw4(event):
            global fcalc, uvw4,hkl4, entry_uvw4, entry_hkl4
            attr_calc(0)
            uvw4 = ToFourIndexDir(uvw)
            hkl4 = ToFourIndexPlane(hkl)
            for i in range(4): entry_uvw4[i].delete(0, 'end')
            for i in range(4): entry_uvw4[i].insert(0, uvw4[i])
            for i in range(4): entry_hkl4[i].delete(0, 'end')
            for i in range(4): entry_hkl4[i].insert(0, hkl4[i])


        def calcuvw(event):
            global fcalc, uvw,hkl, entry_uvw,entry_hkl
            attr_calc(0)
            uvw = ToThreeIndexDir(uvw4)
            hkl = ToThreeIndexPlane(hkl4)
            for i in range(3): entry_uvw[i].delete(0, 'end')
            for i in range(3): entry_uvw[i].insert(0, uvw[i])
            for i in range(3): entry_hkl[i].delete(0, 'end')
            for i in range(3): entry_hkl[i].insert(0, hkl[i])


        def build_fcalc():
            global fcalc, entry_uvw, entry_hkl, entry_uvw4, entry_hkl4
            try: fcalc.destroy()
            except: pass
            fcalc = Frame(frame)
            lt = Label(fcalc, foreground="black")
            lt["text"] = "3 indices"
            lt["height"] = 1
            lt.grid(row=1, column=2, columnspan = 3, sticky= W + E)
            lt = Label(fcalc, foreground="black")
            lt["text"] = "4 indices"
            lt["height"] = 1
            lt.grid(row=1, column=7, columnspan = 3, sticky= W + E)
            lt = Label(fcalc)
            lt["text"] = " [u,v,w] "
            lt["height"] = 1
            lt.grid(row=2, column=1, sticky=E + W)
            entry_uvw = [None for i in range(3)]
            for i in range(3):
                entry_uvw[i] = Entry(fcalc, width=7)
                entry_uvw[i].insert(0, uvw[i])
                entry_uvw[i].bind("<Return>", calcuvw4)
                entry_uvw[i].grid(row=2, column=i + 2, sticky=E + W)

            lt = Label(fcalc, foreground="black")
            lt["text"] = "4 indices"
            lt["height"] = 1
            lt.grid(row=1, column=7, columnspan = 4, sticky=W + E)
            lt = Label(fcalc)
            lt["text"] = " Direction "
            lt["height"] = 1
            lt.grid(row=2, column=1, sticky=E + W)
            entry_uvw4 = [None for i in range(4)]
            for i in range(4):
                entry_uvw4[i] = Entry(fcalc, width=7)
                entry_uvw4[i].insert(0, uvw4[i])
                entry_uvw4[i].bind("<Return>", calcuvw)
                entry_uvw4[i].grid(row=2, column=i + 7, sticky=E + W)
            lt = Label(fcalc, foreground="black")
            lt["text"] = "/3"
            lt.grid(row=2, column=11, sticky=W + E)
            
            lt = Label(fcalc)
            lt["text"] = "-------------------------------------------------------------------------------------"
            lt["height"] = 1
            lt.grid(row=3, column=1, columnspan=10, sticky=E + W)
            lt = Label(fcalc)
            lt["text"] = " Plane "
            lt["height"] = 1
            lt.grid(row=4, column=1, sticky=E + W)
            entry_hkl = [None for i in range(3)]
            for i in range(3):
                entry_hkl[i] = Entry(fcalc, width=7)
                entry_hkl[i].insert(0, hkl[i])
                entry_hkl[i].bind("<Return>", calcuvw4)
                entry_hkl[i].grid(row=4, column=i + 2, sticky=E + W)

            entry_hkl4 = [None for i in range(4)]
            for i in range(4):
                entry_hkl4[i] = Entry(fcalc, width=7)
                entry_hkl4[i].insert(0, hkl4[i])
                entry_hkl4[i].bind("<Return>", calcuvw)
                entry_hkl4[i].grid(row=4, column=i + 7, sticky=E + W)

            fcalc["borderwidth"] = 1
            fcalc["relief"] = SOLID
            fcalc.grid(row=1, column=2, sticky=N)

        build_fcalc()


    def calc_dist():
        global fcalc, uvw, hkl, duvw, dhkl, entry_uvw, entry_hkl

        try: fcalc.destroy()
        except:pass

        uvw = [0, 0, 0]
        hkl = [0, 0, 0]
        duvw = 0
        dhkl = 0

        def attr_calc(event):
            global uvw, hkl
            uvw = []
            for i in range(3):
                uvw.append(string.atoi(entry_uvw[i].get()))
            uvw = array(uvw)
            hkl = []
            for i in range(3):
                hkl.append(string.atoi(entry_hkl[i].get()))
            hkl = array(hkl)

        def calcduvw(event):
            global fcalc, uvw, duvw
            attr_calc(0)
            guvw = vectDir(CRYST, uvw)
            duvw = guvw.length()
            fcalc.destroy()
            build_fcalc()

        def calcdhkl(event):
            global fcalc, hkl, dhkl
            attr_calc(0)
            ghkl = vectRec(CRYST, hkl)
            dhkl = ghkl.length()
            fcalc.destroy()
            build_fcalc()

        def build_fcalc():
            global fcalc, entry_uvw, entry_hkl
            try: fcalc.destroy()
            except: pass
            fcalc = Frame(frame)
            lt = Label(fcalc, foreground="black")
            lt["text"] = "Distances"
            lt["height"] = 1
            lt.grid(row=1, column=1, sticky=W + E)
            lt = Label(fcalc)
            lt["text"] = " [u,v,w] "
            lt["height"] = 1
            lt.grid(row=2, column=1, sticky=E + W)
            lt = Label(fcalc)
            lt["text"] = " Ang. "
            lt["height"] = 1
            lt.grid(row=1, column=6, sticky=E + W)
            lt = Label(fcalc)
            lt["text"] = " 1/Ang. "
            lt["height"] = 1
            lt.grid(row=1, column=7, sticky=E + W)
            entry_uvw = [None for i in range(3)]
            for i in range(3):
                entry_uvw[i] = Entry(fcalc, width=7)
                entry_uvw[i].insert(0, uvw[i])
                entry_uvw[i].bind("<Return>", calcduvw)
                entry_uvw[i].grid(row=2, column=i + 2, sticky=E + W)
            lt = Label(fcalc)
            lt["text"] = "-------------------------------------------------------------------------------------"
            lt["height"] = 1
            lt.grid(row=3, column=1, columnspan=7, sticky=E + W)
            lt = Label(fcalc)
            lt["text"] = " (h,k,l) "
            lt["height"] = 1
            lt.grid(row=4, column=1, sticky=E + W)
            entry_hkl = [None for i in range(3)]
            for i in range(3):
                entry_hkl[i] = Entry(fcalc, width=7)
                entry_hkl[i].insert(0, hkl[i])
                entry_hkl[i].bind("<Return>", calcdhkl)
                entry_hkl[i].grid(row=4, column=i + 2, sticky=E + W)
            entry_duvw = [None for j in range(2)]
            entry_dhkl = [None for j in range(2)]
            for j in range(2):
                entry_duvw[j] = Entry(fcalc, width=7)
                try:
                    if j == 0: entry_duvw[j].insert(0, duvw)
                    else: entry_duvw[j].insert(0, 1. / duvw)
                except: entry_duvw[j].insert(0, 0)
                entry_duvw[j].grid(row=2, column=6 + j, sticky=E + W)
                entry_duvw[j]["background"] = colorg
            for j in range(2):
                entry_dhkl[j] = Entry(fcalc, width=7)
                try:
                    if j == 1: entry_dhkl[j].insert(0, dhkl)
                    else: entry_dhkl[j].insert(0, 1. / dhkl)
                except: entry_dhkl[j].insert(0, 0)
                entry_dhkl[j].grid(row=4, column=6 + j, sticky=E + W)
                entry_dhkl[j]["background"] = colorg
            fcalc["borderwidth"] = 1
            fcalc["relief"] = SOLID
            fcalc.grid(row=1, column=2, sticky=N)

        build_fcalc()

    def calc_angle():

        global fcalc, uvw1, u, v, w2, hkl1, hkl2, auvw, ahkl, entry_uvw1, entry_uvw2, entry_hkl1, entry_hkl2
        colorg = "grey90"

        uvw1 = [0, 0, 0]
        uvw2 = [0, 0, 0]
        hkl1 = [0, 0, 0]
        hkl2 = [0, 0, 0]
        auvw = 0
        ahkl = 0

        try: fcalc.destroy()
        except:pass

        def attr_calc(event):
            global uvw1, hkl1, uvw2, hkl2
            uvw1 = []
            for i in range(3):
                uvw1.append(string.atoi(entry_uvw1[i].get()))
            uvw1 = array(uvw1)
            hkl1 = []
            for i in range(3):
                hkl1.append(string.atoi(entry_hkl1[i].get()))
            hkl1 = array(hkl1)
            uvw2 = []
            for i in range(3):
                uvw2.append(string.atoi(entry_uvw2[i].get()))
            uvw2 = array(uvw2)
            hkl2 = []
            for i in range(3):
                hkl2.append(string.atoi(entry_hkl2[i].get()))
            hkl2 = array(hkl2)

        def calcauvw(event):
            global fcalc, uvw1, uvw2, auvw
            attr_calc(0)
            auvw = angle(uvw1, uvw2, CRYST.metricTensDir)
            fcalc.destroy()
            build_fcalc()

        def calcahkl(event):
            global fcalc, hkl1, hkl2, ahkl
            attr_calc(0)
            ahkl = angle(hkl1, hkl2, CRYST.metricTensRec)
            fcalc.destroy()
            build_fcalc()

        def build_fcalc():
            global fcalc, entry_uvw1, entry_uvw2, entry_hkl1, entry_hkl2
            try: fcalc.destroy()
            except: pass
            fcalc = Frame(frame)
            lt = Label(fcalc, foreground="black")
            lt["text"] = "Angle"
            lt["height"] = 1
            lt.grid(row=1, column=1, sticky=W + E)
            lt["text"] = " Degrees "
            lt["height"] = 1
            lt.grid(row=1, column=6, sticky=E + W)
            lt = Label(fcalc)
            lt["text"] = " Radians "
            lt["height"] = 1
            lt.grid(row=1, column=7, sticky=E + W)
            lt = Label(fcalc)
            lt["text"] = " [u,v,w] 1"
            lt["height"] = 1
            lt.grid(row=2, column=1, sticky=E + W)
            lt = Label(fcalc)
            entry_uvw1 = [None for i in range(3)]
            for i in range(3):
                entry_uvw1[i] = Entry(fcalc, width=7)
                entry_uvw1[i].insert(0, uvw1[i])
                entry_uvw1[i].bind("<Return>", calcauvw)
                entry_uvw1[i].grid(row=2, column=i + 2, sticky=E + W)
            lt = Label(fcalc)
            lt["text"] = " [u,v,w] 2"
            lt["height"] = 1
            lt.grid(row=3, column=1, sticky=E + W)
            lt = Label(fcalc)
            entry_uvw2 = [None for i in range(3)]
            for i in range(3):
                entry_uvw2[i] = Entry(fcalc, width=7)
                entry_uvw2[i].insert(0, uvw2[i])
                entry_uvw2[i].bind("<Return>", calcauvw)
                entry_uvw2[i].grid(row=3, column=i + 2, sticky=E + W)
            lt = Label(fcalc)
            lt["text"] = "-------------------------------------------------------------------------------------"
            lt["height"] = 1
            lt.grid(row=4, column=1, columnspan=7, sticky=E + W)
            lt = Label(fcalc)
            lt["text"] = " (h,k,l) 1"
            lt["height"] = 1
            lt.grid(row=5, column=1, sticky=E + W)
            entry_hkl1 = [None for i in range(3)]
            for i in range(3):
                entry_hkl1[i] = Entry(fcalc, width=7)
                entry_hkl1[i].insert(0, hkl1[i])
                entry_hkl1[i].bind("<Return>", calcahkl)
                entry_hkl1[i].grid(row=5, column=i + 2, sticky=E + W)
            lt = Label(fcalc)
            lt["text"] = " (h,k,l) 2"
            lt["height"] = 1
            lt.grid(row=6, column=1, sticky=E + W)
            entry_hkl2 = [None for i in range(3)]
            for i in range(3):
                entry_hkl2[i] = Entry(fcalc, width=7)
                entry_hkl2[i].insert(0, hkl2[i])
                entry_hkl2[i].bind("<Return>", calcahkl)
                entry_hkl2[i].grid(row=6, column=i + 2, sticky=E + W)
            entry_auvw = [None for j in range(2)]
            entry_ahkl = [None for j in range(2)]
            for j in range(2):
                entry_auvw[j] = Entry(fcalc, width=7)
                if j == 0: entry_auvw[j].insert(0, auvw)
                else: entry_auvw[j].insert(0, auvw * Degree)
                entry_auvw[j].grid(row=3, column=6 + j, sticky=E + W)
                entry_auvw[j]["background"] = colorg
            for j in range(2):
                entry_ahkl[j] = Entry(fcalc, width=7)
                if j == 0: entry_ahkl[j].insert(0, ahkl)
                else: entry_ahkl[j].insert(0, ahkl * Degree)
                entry_ahkl[j].grid(row=6, column=6 + j, sticky=E + W)
                entry_ahkl[j]["background"] = colorg
            fcalc["borderwidth"] = 1
            fcalc["relief"] = SOLID
            fcalc.grid(row=1, column=2, sticky=N)

        build_fcalc()


    def calc_normal():

        global fcalc, uvw1, uvw2, hkl1, hkl2, ratiores, entry_uvw1, entry_uvw2, entry_hkl1, entry_hkl2, entry_rat
        colorg = "grey90"

        uvw1 = [0, 0, 0]
        uvw2 = [0, 0, 0]
        hkl1 = [0, 0, 0]
        hkl2 = [0, 0, 0]
        ratiores = 0.01

        try: fcalc.destroy()
        except:pass

        def attr_calc(event):
            global uvw1, hkl1, uvw2, hkl2, ratiores
            ratiores = string.atof(entry_rat.get())
            uvw1 = []
            for i in range(3):
                uvw1.append(string.atoi(entry_uvw1[i].get()))
            uvw1 = array(uvw1)
            hkl1 = IntegriseAxe(dot(CRYST.metricTensDir,uvw1),ratiores)
            hkl1 = array(hkl1)
            hkl2 = []
            for i in range(3):
                hkl2.append(string.atoi(entry_hkl2[i].get()))
            hkl2 = array(hkl2)
            uvw2 = IntegriseAxe(dot(CRYST.metricTensRec,hkl2),ratiores)
            uvw2 = array(uvw2)

        def calc0(event):
            global fcalc, uvw1, uvw2, hkl1, hkl2
            attr_calc(0)
            fcalc.destroy()
            build_fcalc()

        def build_fcalc():
            global fcalc, entry_uvw1, entry_uvw2, entry_hkl1, entry_hkl2, entry_rat
            try: fcalc.destroy()
            except: pass
            fcalc = Frame(frame)
            lt = Label(fcalc, foreground="black")
            lt["text"] = " resolution on rational vector"
            lt["height"] = 1
            lt.grid(row=1, column=1, sticky=E + W)
            entry_rat = Entry(fcalc, width=7)
            entry_rat.insert(0, ratiores)
            entry_rat.bind("<Return>", calc0)
            entry_rat.grid(row=1, column=2, sticky=E + W)          
            lt = Label(fcalc, foreground="black")
            lt["text"] = "The plane normal to [u,v,w]"
            lt["height"] = 1
            lt.grid(row=2, column=1, sticky=W)
            lt = Label(fcalc)
            entry_uvw1 = [None for i in range(3)]
            for i in range(3):
                entry_uvw1[i] = Entry(fcalc, width=7)
                entry_uvw1[i].insert(0, uvw1[i])
                entry_uvw1[i].bind("<Return>", calc0)
                entry_uvw1[i].grid(row=2, column=i + 2, sticky=E + W)
            lt = Label(fcalc)
            lt["text"] = " is (h,k,l)="
            lt["height"] = 1
            lt.grid(row=2, column=5, sticky=E + W)
            entry_hkl1 = [None for i in range(3)]
            for i in range(3):
                entry_hkl1[i] = Entry(fcalc, width=7)
                entry_hkl1[i].insert(0, hkl1[i])
                entry_hkl1[i].bind("<Return>", calc0)
                entry_hkl1[i].grid(row=2, column=i + 6, sticky=E + W)
            lt = Label(fcalc)
            lt["text"] = "-------------------------------------------------------------------------------------"
            lt["height"] = 1
            lt.grid(row=4, column=1, columnspan=7, sticky=E + W)
            lt = Label(fcalc)
            lt["text"] = "The direction normal to (h,k,l)"
            lt["height"] = 1
            lt.grid(row=4, column=1, sticky=W)
            lt = Label(fcalc)
            entry_hkl2 = [None for i in range(3)]
            for i in range(3):
                entry_hkl2[i] = Entry(fcalc, width=7)
                entry_hkl2[i].insert(0, hkl2[i])
                entry_hkl2[i].bind("<Return>", calc0)
                entry_hkl2[i].grid(row=4, column=i + 2, sticky=E + W)
            lt = Label(fcalc)
            lt["text"] = " is [u,v,w]="
            lt["height"] = 1
            lt.grid(row=4, column=5, sticky=E + W)
            entry_uvw2 = [None for i in range(3)]
            for i in range(3):
                entry_uvw2[i] = Entry(fcalc, width=7)
                entry_uvw2[i].insert(0, uvw2[i])
                entry_uvw2[i].bind("<Return>", calc0)
                entry_uvw2[i].grid(row=4, column=i + 6, sticky=E + W)
            fcalc["borderwidth"] = 1
            fcalc["relief"] = SOLID
            fcalc.grid(row=1, column=2, sticky=N)

        build_fcalc()


    def calc_listduvw():
        global fcalc, entry_distmax, entry_dmin, distmax, SymFlag

        SymFlag = 1

        try: fcalc.destroy()
        except:pass

        def attr_distmax(event):
            global fcalc, distmax
            distmax = string.atof(entry_distmax.get())
            fcalc.destroy()
            build_fcalc()

        def attr_sym():
            global fcalc, SymFlag
            SymFlag = (SymFlag + 1) % 2
            fcalc.destroy()
            build_fcalc()

        def build_fcalc():
            global fcalc, entry_distmax, entry_dmin, SymFlag

            esp = '        '
            esp2 = '  '
            try: fcalc.destroy()
            except: pass
            fcalc = Frame(frame)
            l = Label(fcalc, text="______ List of reticular distances _____")
            l.grid (row=0, column=1, columnspan=4)
            fflag = Frame(fcalc)
            Checkbutton (fflag, text="With/without symmetries", variable=tkSym, command=attr_sym).grid (row=2, column=1, columnspan=6)
        
            lt = Label(fflag, foreground="black", text=" dist. max = ").grid(row=3, column=4, sticky= W + E)
            entry_distmax = Entry(fflag, width=4)
            f = str(distmax)
            f = f[:f.find(".") + 2]
            entry_distmax.insert(0, f)
            entry_distmax.bind("<Return>", attr_distmax)
            entry_distmax.grid(row=3, column=5, sticky=E + W)
            fflag.grid (row=1, column=1, columnspan=4)
            lt = Label(fcalc, foreground="black", text="", height=1).grid(row=4, column=1, columnspan=2, sticky=W)
            frame2 = Frame(fcalc, width=50, borderwidth=4)
            lt = Label(frame2, foreground="black", text=" duvw " + esp + "u v w" + esp + "  Intensity", height=1).grid(row=1, column=1, columnspan=2, sticky=W+E)
            Listd = Listbox(frame2, width=35, height=30)
            Listd.grid (row=2, column=2)
            s = Scrollbar(frame2)
            s.grid(row=2, column=1, sticky="NS")
            s.config(command=Listd.yview)
            Listd.config(yscrollcommand=s.set)
            for uvw in CRYST.listduvw(SymFlag, distmax):
                d = str(uvw[0])[:5]
                while len(d) <= 5:
                    d = d + '0'
                text = str("  " + d + esp + esp2 + str(uvw[1]) + esp2 + str(uvw[2]) + esp2 + str(uvw[3]) + esp + str(round(uvw[4], 4)))
                #print text
                Listd.insert(END, text)
                text = "---------------------------------------------------------------------------"
                Listd.insert(END, text)
            frame2.grid (row=5, column=1, columnspan=4)
            fcalc["borderwidth"] = 1
            fcalc["relief"] = SOLID
            fcalc.grid(row=1, column=2, sticky=N)

        build_fcalc()
        
    #===============================================================================================================
    # ====================================== WEAK TWINS ============================================================
    #===============================================================================================================
    def calc_weakhkl():
        global fcalc, entry_distmax, entry_axePi, distmax, errdistmax, errangmax, axePi, listweakplanes, listweakplanesAxes,rankweak, ModeWeak
        global twin,twinold, uvwmax, listtoshow, sgmax

        v = sys.version[:3]
        versionP = "Python" + str(reduce(lambda x,y:x+y, v.split(".")))
        try:
            dirtwin = "c:/"+versionP+"/ElectronMicroscopy_Data\Results"
            os.chdir(dirtwin)
        except: pass

        EtatRun = StringVar()
        EtatRun.set("")

        SymFlag = 1
        rankweak = 0
        errdistmax = 5 # %
        errangmax = 5 # degree
        axePi = [0,0,0]
        listweakplanes = []
        listweakplanesAxes=[]
        ModeWeak = 1 # 1 = "Pi"
        twin,twinold = -1,-1
        uvwmax = 1 # index max for the axis of the weak plane
        sgmax = 0.5 #generalized strain max
        listtoshow=[]

        try: fcalc.destroy()
        except:pass

        def showorient(m,nbl=6): # copy past from that of regular twins
            ffInfoTwin = open("ListInfoTwin.txt", 'a') #on continue la list C,F, maintenant avec T
            deg_u = '\u00ba'
            deg = deg_u.encode('utf-8')
            OrientMat = m
            text = "Orientation Matrix T (parent => twin) " 
            TwinBox.insert(END, text)
            ffInfoTwin.write("\n"+text+"\n")
            text = tabify("  ",nbl) + tabify(cut(OrientMat[0,0]))  + tabify(cut(OrientMat[0,1]))  + tabify(cut(OrientMat[0,2]))
            TwinBox.insert(END, text)
            ffInfoTwin.write(text+"\n")
            text = tabify("  ",nbl)  + tabify(cut(OrientMat[1,0])) + tabify(cut(OrientMat[1,1]))  + tabify(cut(OrientMat[1,2]))
            TwinBox.insert(END, text)
            ffInfoTwin.write(text+"\n")
            text = tabify("  ",nbl) + tabify(cut(OrientMat[2,0])) +  tabify(cut(OrientMat[2,1])) +  tabify(cut(OrientMat[2,2]))
            TwinBox.insert(END, text)
            ffInfoTwin.write(text+"\n"+"\n")
            TwinBox.insert(END, " ")          
            if (CRYST.syst == "Hexagonal" or CRYST.syst == "Trigonal"):
                text = "Angle between the c axes = " 
                c1 = array([0,0,1])
                c2 = dot(OrientMat,c1)
                a12 = angleVect(CRYST, vectDir(CRYST,c1),vectDir(CRYST,c2))
                a12b = 180-a12
                TwinBox.insert(END, text + cut(a12) + " Deg. (complement = " + cut(a12b) + " Deg.)" )
                ffInfoTwin.write(text + cut(a12) + " Deg. (complement = " + cut(a12b) + " Deg.)" )
                TwinBox.insert(END, " ")
                ffInfoTwin.write("\n")
            TwinBox.insert(END, "List of equivalent rotations (ang/axis)")
            ffInfoTwin.write( "List of equivalent rotations (ang/axis)"+"\n")
            if CRYST.pg == CRYST.holopg():
                if det(OrientMat)<0: 
                    OrientMat = dot(-1,OrientMat)
                    textpol= " -1x \n"
                else:
                    textpol= " "
                for roti in ListEquivRot(OrientMat, CRYST.sym(),CRYST.structTensDir):
                    ang,axis,mk = roti
                    esp = ' ' 
                    if len(axis)>0:
                        if (CRYST.syst != "Hexagonal" and CRYST.syst != "Trigonal"):
                            try: text = tabify(str(cut(ang)) + deg + "   " )+  tabify(" Axis = " + esp + str(axis[0]) + esp + str(axis[1]) + esp + str(axis[2]))
                            except: text = ''
                        else:
                            try:
                                axis4 = ToFourIndexDir(axis)
                                text = tabify(str(cut(ang)) + deg + "   " )+  tabify(" Axis = [" +  str(axis[0]) + esp + str(axis[1]) + esp + str(axis[2]) + "]")
                                text2 = tabify(" = 1/3 [" + str(axis4[0]) + esp + str(axis4[1]) + esp + str(axis4[2]) + esp + str(axis4[3]) + "]")
                                text = text + text2
                            except: text = ''
                        
                    else: text = "Identity"
                    if text:
                        TwinBox.insert(END, textpol+text)
                        ffInfoTwin.write(textpol+text+"\n")
            else:
                #TwinBox.insert(END, "There are different possible orientations because the crystal is not holohedric")
                # calculate the variants by coset decomposition HoloPG/PG
                print("CRYST.pg, CRYST.holopg() dans else", CRYST.pg, CRYST.holopg())# ICI CORRIGER ICI
                holo = CRYST.holosym()
                QuotientG = []
                for sholopg in CRYST.holosym():
                    coset = []
                    for s in CRYST.sym():
                        coset.append(dot(sholopg,s))
                    flag = 1
                    for Q in QuotientG:
                        if Inter(coset,Q):
                            flag=0
                            break
                    if flag: QuotientG.append(coset)
                #print "nb of variants holo => mero (simple cosets) = ", len(QuotientG)
                # calculate the misorientations between the twins by coset.T.coset
                listT = []
                for coseti in QuotientG:
                    si = coseti[0]
                    for cosetj in QuotientG:
                        sj = cosetj[0]
                        newT = dot(dot(inv(si),OrientMat),sj)
                        listT.append(newT)
                listTred = []
                invT = inv(OrientMat)
                for newT in listT :
                    if not IsInGroup(dot(invT,newT),CRYST.sym()): listTred.append(newT)                    
                text = "There are "+ str(len(listTred)) +" twins by double-cosets from merohedry <= holohedry / holohedry => merohedry"
                TwinBox.insert(END, text)
                ffInfoTwin.write(text+"\n")
                TwinBox.insert(END, " ")
                indt = 0
                for newOrientMat in listTred:
                    indt+=1
                    TwinBox.insert(END, "rotation list for twin "+str(indt))
                    ffInfoTwin.write("rotation list for twin "+str(indt)+"\n")
                    if det(newOrientMat)<0:
                        newOrientMat = -newOrientMat
                        textpol = "-1 x "
                    else: textpol = "  "
                    for roti in ListEquivRot(newOrientMat, CRYST.sym(),CRYST.structTensDir):
                        ang,axis,mk = roti
                        esp = ' ' 
                        if len(axis)>0:
                            if (CRYST.syst != "Hexagonal" and CRYST.syst != "Trigonal"):
                                try: text = tabify(str(cut(ang)) + deg )+  tabify(" Axis = [" + str(axis[0]) + esp + str(axis[1]) + esp + str(axis[2]))
                                except: text = ''
                            else:
                                try:
                                    axis4 = ToFourIndexDir(axis)
                                    text = tabify(str(cut(ang)) + deg + "   " )+  tabify(" Axis = [" +  str(axis[0]) + esp + str(axis[1]) + esp + str(axis[2]) + "]")
                                    text2 = tabify("  = 1/3 [" + str(axis4[0]) + esp + str(axis4[1]) + esp + str(axis4[2]) + esp + str(axis4[3]) + "]")
                                    text = text + text2
                                except: text = ''
                            
                        else: text = "Identity"
                        if text:
                            TwinBox.insert(END, textpol + text)
                            ffInfoTwin.write(textpol + text+"\n")
                    TwinBox.insert(END, " ")
                ffInfoTwin.close()

        def runtwin():
            if ModeWeak==1: do_calc_180weak() 
            elif ModeWeak==2: do_calc_Axeweak()
            else: do_calc_AxeweakAll()
            build_fcalc() 

        def attr_par(event):
            global fcalc, distmax, errdistmax, errangmax, axePi, uvwmax, entry_uvwmax,sgmax
            distmax = string.atof(entry_distmax.get())
            errdistmax = string.atof(entry_errdistmax.get())
            errangmax = string.atof(entry_errangmax.get())
            try: sgmax = string.atof(entry_sgmax.get())
            except: pass
            if ModeWeak==1 or ModeWeak==2:
                axePi=[]
                for i in range(3):
                    axePi.append(string.atoi(entry_axePi[i].get()))
                if axePi==[0,0,0] and ModeWeak==2: uvwmax = string.atoi(entry_uvwmax.get())
                else:uvwmax = max(axePi)
            runtwin()

        def WeakMode1(event):
            global ModeWeak, rankweak, fcalc 
            ModeWeak= 1
            rankweak = 0
            try: fcalc.destroy()
            except: pass
            build_fcalc()
            
        def WeakMode2(event):
            global ModeWeak, rankweak, fcalc 
            ModeWeak= 2
            rankweak = 0
            try: fcalc.destroy()
            except: pass
            build_fcalc()
            
        def WeakMode3(event):
            global ModeWeak, rankweak, fcacl
            ModeWeak = 3
            rankweak = 3
            try: fcalc.destroy()
            except: pass
            build_fcalc()

        def rank_errdist(event):
            global rankweak
            if rankweak == 0 or rankweak == 3 or rankweak == 4: rankweak = 1
            else: rankweak = 0
            build_fcalc()

        def rank_errang(event):
            global rankweak
            if rankweak == 0 or rankweak == 3 or rankweak == 4: rankweak = 1
            else: rankweak = 0
            build_fcalc()

        def rank_errs(event):
            global rankweak
            if rankweak == 3 or rankweak == 0 or rankweak == 1: rankweak = 4
            else: rankweak = 3
            build_fcalc()

        def rank_errp(event):
            global rankweak
            if rankweak == 4 or rankweak == 0 or rankweak == 1: rankweak = 3
            else: rankweak = 4
            build_fcalc()

        def do_calc_listdir(): # list of directions of close length uvwi,uvwgj    
            listcoupledir = []
            listuvw = CRYST.listduvw(SymFlag, distmax)
            for i in range(len(listuvw)):
                uvwi = listuvw[i][1:4]
                auvwi = array(uvwi)
                di = listuvw[i][0]
                #dir1 = vectDir(CRYST,uvw1[1:4])
                for j in range(i+1,len(listuvw)): # 16 August 2020 I tried i instead of i+1, to be able to capture the cases dir => -dir as in pericline cases, without success
                    auvwj = array(listuvw[j][1:4])
                    dj = listuvw[j][0]
                    #errd = min(abs(di/dj-1),abs(di/(2*dj)-1),abs(dj/(2*di)-1)) #autorise les doubles, a voir si besoin triples
                    errd = abs(di/dj-1)
                    #errd = abs(di/dj-1)
                    if errd<float(errdistmax)/100:
                        fact=1
                        if abs(di/(2*dj)-1)<float(errdistmax)/100: uvwj = (2*auvwj).tolist()
                        elif abs(dj/(2*di)-1)<float(errdistmax)/100: uvwi = (2*auvwi).tolist()                   
                        planij = ReduceAxe(cross(auvwi,auvwj).tolist())
                        #angij = angle(array(uvwi), array(axePi), CRYST.metricTensDir)
                        if planij!=[0,0,0]:
                            #listcoupledir.append([di,errd,uvwi.tolist(),uvwj.tolist(),planij])
                            for gj in CRYST.sym():
                                flagplan = 1
                                auvwgj = dot(gj,auvwj)
                                uvwgj = auvwgj.tolist()
                                uvwigj = dot(gj,auvwi).tolist()
                                plangij = ReduceAxe(IntegriseAxe(cross(auvwi,auvwgj)).tolist())
                                if uvwigj!= uvwi: 
                                    listcoupledir.append([di,errd,uvwi,uvwgj,plangij])

            return listcoupledir

        def ClosestReticular(plan,OZ1,OH1,U,V,mUV,q): # q peut etre 1,2,3, .. ou -1, -2, ..
            OH = q*OH1
            OZ = q*OZ1
            ZH = -OZ+OH
            ZH_inUV = dot(mUV,ZH)
            nU,rU = ft_min(ZH_inUV[0])
            nV,rV = ft_min(ZH_inUV[1])
            ZA = (nU*U + nV*V) # A is the closest point from H by coordinate, not necessarily by distance
            sU = sign(rU)
            if sU ==0: sU = 1
            sV = sign(rV)
            if sV ==0: sV = 1
            ZB = ZA + sU*U
            ZC = ZA + sV*V
            ZD = ZA + sU*U + sV*V
            OA,OB,OC,OD = OZ+ZA, OZ+ZB, OZ+ZC, OZ+ZD
            return [OA, OB, OC, OD]

        def Reticular(plan,OZ1,OH1,U,V,mUV,q,radius=1):
            OH = q*OH1
            OZ = q*OZ1
            ZH = -OZ+OH
            ZH_inUV = dot(mUV,ZH)
            nU,rU = ft_min(ZH_inUV[0])
            nV,rV = ft_min(ZH_inUV[1])
            ZA = (nU*U + nV*V) # A is the closest point from H by coordinate, not necessarily by distance
            OA = OZ+ZA
            listZP=[]
            for i in range(-radius,radius+1):
                for j in range(-radius,radius+1):  
                    listZP.append(OA + i*U + j*V)
            return listZP
            
        def findWeakBasis(axe,vi,plani,vj,planj,qmax=2):      
            def take0(elem):
                return elem[0]
            hi,ki,li = plani
            hj,kj,lj = planj
            OZi,Ui,Vi = Bezout3D(hi,ki,li) # OZ = [u,v,w] a solution of uh+kv+vw = 1, U and V two vectors such that U.(h,k,l)=0 and V.(h,k,l)=0
            OZj,Uj,Vj = Bezout3D(hj,kj,lj)
            deti = det(array([axe,vi,OZi]))
            detj = det(array([axe,vj,OZj]))
            ratqi,ratqj = 1,1
            if abs(deti) == abs(detj):
                ratqi,ratqj = 1,1
            elif abs(deti)>abs(detj):
                ratqi,ratqj = 1,int(deti/detj)
            else:
                ratqi,ratqj = int(detj/deti),1
            if deti*detj<0: ratqj= -ratqj
            
            mUVi =  LeftInv(transpose(array([Ui,Vi])))
            mUVj =  LeftInv(transpose(array([Uj,Vj])))
          
            apDi = dot(CRYST.metricTensRec,plani)
            dhkli2 = (1./dot(plani,apDi)) # dhkl square
            OHi = dhkli2*apDi # vector normal to the plane hkl such that its norm = dhkl
            ni = sqrt(dhkli2)*apDi # same but norm = 1
            apDj = dot(CRYST.metricTensRec,planj)
            dhklj2 = (1./dot(planj,apDj)) # dhkl square
            OHj = dhklj2*apDj # vector normal to the plane hkl such that its norm = dhkl
            nj = sqrt(dhklj2)*apDj # same but norm = 1
            mi = dot(CRYST.metricTensRec,cross(axe,ni)) # cest la normale a laxe et a la normale OHi
            mj = dot(CRYST.metricTensRec,cross(axe,nj)) # cest la normale a laxe et a la normale OHj
            BasisiT = transpose(array([axe,ni, mi]))
            BasisjTup1 = transpose(array([axe,nj, mj]))
            BasisjTdown1 = transpose(array([axe,-nj, mj])) # car il y a deux facons de tourner le plan a 180 deg lune de lautre
            #BasisjTup2 = transpose(array([array(axe),nj, -mj]))
            BasisjTdown2 = transpose(array([array(axe),-nj, -mj]))
            matorientup1 = dot(BasisiT,inv(BasisjTup1))
            matorientdown1 = dot(BasisiT,inv(BasisjTdown1))
            #matorientup2 = dot(BasisiT,inv(BasisjTup2))
            matorientdown2 = dot(BasisiT,inv(BasisjTdown2)) # les up 2 et down2 ajoute le 7 sept

            flag = 0
            listOA,listOP = [],[]
            listresW=[]
            M = CRYST.metricTensDir
            ME = CRYST.metricTensRec
            listreject=[]
            for q in range(0,qmax+1):
                #print "============",q,q*ratqi,q*ratqj," ========="
                if q!=0:
                    listOA = ClosestReticular(plani,OZi,OHi,Ui,Vi,mUVi, q*ratqi)
                    listOP = ClosestReticular(planj,OZj,OHj,Uj,Vj,mUVj, q*ratqj)               
                    for OAi in listOA:
                        Basisi = transpose(array([axe,array(vi),OAi]))
                        detiA = det(Basisi)
                        #if flag: break # je suppose qu une seule macle faible est possible par plan faible
                        for OPj in listOP:
                            BasisjR1 = transpose(array([array(axe),array(vj),OPj]))
                            #BasisjR2 = transpose(array([array(axe),-array(vj),OPj])) 
                            BasisjL1 = transpose(array([array(axe),array(vj),-OPj]))
                            #BasisjL2 = transpose(array([array(axe),-array(vj),-OPj]))
                            lbas = [BasisjR1,BasisjL1] # 7 sept   BasisjR1,BasisjL1
                            for i in range(len(lbas)): 
                                Basisj = lbas[i] 
                                if i==0: sign = 1
                                else: sign = -1
                                detjP = det(Basisj)
                                #print "OAi OPj deti detj = ", OAi.tolist(), OPj.tolist(), deti, detj
                                if abs(detiA)==abs(detjP) and detiA!=0:
                                    for matorient in [matorientup1,matorientdown1,matorientdown2]: #for matorient in [matorientup1,,matorientdown1]: # down2 et up 2 le 7 sept
                                        matcor = dot(Basisj,inv(Basisi)) 
                                        F = dot(matorient,matcor)
                                        #sg = ShearBevisCrocker(matcor,CRYST) # cannot be used because insensitive to rotations!!
                                        sg = ShearMyFormula(F,CRYST)
                                        if sg<sgmax:
                                            if [OAi.tolist(),dot(sign,OPj).tolist()] not in listreject:
                                                listresW.append([sg,OAi,dot(sign,OPj),matorient,matcor,F]) # ICI
                                                listreject.append([OAi.tolist(),(dot(sign,OPj)).tolist()]) 
                                                #print "vi,vj,OAi,dot(sign,OPj)",vi,vj,OAi,dot(sign,OPj)
            if listresW:
                listresW.sort(key = take0)
                return listresW 
            else: return []


        def listequiv(u,axe):
            listres=[]
            listplan=[]
            for g in CRYST.sym():
                gu=dot(g,u).tolist()
                plang = cross(array(gu),axe).tolist()
                flag=1
                if plang!=[0,0,0]:
                    for plan in listplan:
                        if plang == plan:
                            flag=0
                            break
                    if flag:
                        if gu not in listres: listres.append(gu)
                        for s in CRYST.symrec():
                            listplan.append(dot(s,array(plang)).tolist())
            return listres

        def listequiv2(u):
            listres=[]
            for g in CRYST.sym():
                gu=dot(g,u).tolist()
                if gu not in listres: listres.append(gu)
            return listres
                                        
                                        
        def do_calc_180weak():
            global listweakplanes
            listcoupledir = do_calc_listdir() # couple de directions dont les longueurs sont proches
            listweakplanes0=[]
##            if axePi!=[0,0,0]:
##                listaxePi = []
##                for g in CRYST.sym():
##                    listaxePi.append(dot(g,array(axePi)).tolist())
            for c in listcoupledir:    
                di,errd, uvwi,uvwj,planij = c
                uvwSum =  IntegriseAxe(array(uvwi) + array(uvwj))
                uvwDiff = IntegriseAxe(array(uvwi) - array(uvwj))
                auvwi = angle(uvwi, uvwSum, CRYST.metricTensDir)
                auvwj = angle(uvwj, uvwSum, CRYST.metricTensDir)
                buvwi = angle(uvwi, uvwDiff, CRYST.metricTensDir)
                buvwj = angle(uvwj, uvwDiff, CRYST.metricTensDir)
                erra = abs(auvwi+auvwj)
                errb = abs(buvwi+buvwj)
                erra = min(erra, abs(erra+180), abs(erra-180))
                errb = min(errb,abs(errb-180),abs(errb+180))
                if axePi==[0,0,0]:
                    if erra <errangmax:
                        listweakplanes0.append([uvwi,uvwj,ReduceAxe(uvwSum.tolist()),di,errd,abs(auvwi),erra,planij])
                    if errb <errangmax:
                        listweakplanes0.append([uvwi,uvwj,ReduceAxe(uvwDiff.tolist()),di,errd,abs(auvwi),errb,planij])
                else:
                    if erra <errangmax:
                        axe = ReduceAxe(uvwSum.tolist())
                        if axe == axePi: listweakplanes0.append([uvwi,uvwj,axe,di,errd,abs(auvwi),erra,planij])
                    if errb <errangmax:
                        axe = ReduceAxe(uvwDiff.tolist())
                        if axe == axePi: listweakplanes0.append([uvwi,uvwj,axe,di,errd,abs(auvwi),errb,planij])
            listweakplanes = []
            listtokeep = []
            for c in listweakplanes0:
                uvwi,uvwj,axe,di,errd,auvwi,errb,planij = c
                if [axe,planij] not in listtokeep:
                    listweakplanes.append(c)
                    for g in CRYST.sym():
                        gplanij = dot(transpose(g),array(planij)).tolist()
                        gaxe = dot(g,array(axe)).tolist()
                        gaxem = dot(g,-array(axe)).tolist()
##                        guvwi = dot(g,array(uvwi)).tolist()
##                        guvwj = dot(g,array(uvwj)).tolist()
                        listtokeep.append([gaxe,gplanij])
                        listtokeep.append([gaxem,gplanij])
                    
                    
        def do_calc_AxeweakInfo(axe):
            global fcalc
            # attention listweakplanesAxes pas en variable globale mais juste locale
            listweakplanesAxes0 = []
            listcoupledir = do_calc_listdir() # couple de directions dont les longueurs sont proches IL FAUDRAIT LES CALCULER UNE FOIS POUR TOUTE
            listpairplan=[]
            ComptMax = len(listcoupledir)
            compt = 0
            for c in listcoupledir:
                if ModeWeak==2: 
                    perc = (100*compt) / ComptMax
                    text = str(" " + str(perc) + " %")
                    EtatRun.set(text)
                    fcalc.update()
                    compt+=1
                di,errd, ui,uj,planij = c
                for uvwi in listequiv(ui,axe): #25 juin
                #for uvwi in [ui]:
                    #for uvwj in listequiv(uj,axe): # 25 juin
                    for uvwj in listequiv2(uj):   
                        auvwi = angle(array(uvwi), array(axe), CRYST.metricTensDir)
                        auvwj = angle(array(uvwj), array(axe), CRYST.metricTensDir)
                        erra = abs(auvwi+auvwj)
                        errb = abs(auvwi-auvwj)
                        #erra = min(erra, abs(erra-180),abs(erra+180),errb, abs(errb-180),abs(errb+180))
                        erra = min(erra,errb)
                        if erra <errangmax:
                            plani = cross(uvwi,array(axe))
                            planiR = ReduceAxe(IntegriseAxe(plani).tolist())
                            planj = cross(uvwj,array(axe))
                            planjR = ReduceAxe(IntegriseAxe(planj).tolist())
                            angij = angle(plani, planj, CRYST.metricTensRec)
                            flag=1
                            for pplan in listpairplan: # une second tri sera fait dans la section suivante
                                conda = abs(abs(angij)-abs(pplan[0]))<0.0001 
                                cond1 = [uvwi,planiR,uvwj,planjR]==pplan[1:] or [uvwj,planjR,uvwi,planiR]==pplan[1:]
                                cond2 = (-array([uvwi,planiR,uvwj,planjR])).tolist()==pplan[1:] or (-array([uvwj,planjR,uvwi,planiR])).tolist()==pplan[1:]
                                if  conda and (cond1 or cond2): # on finira le travail avec les symmetries dans la section suivante
                                    flag=0
                                    break
                            if flag: 
                                infotwin = findWeakBasis(axe,uvwi,planiR,uvwj,planjR)
                                # infotwin = [sg,OAi,OPj,matorient,matcor,F]
                                if infotwin:
                                    for elttwin in infotwin:
                                        listweakplanesAxes0.append([axe,uvwi,planiR,uvwj,planjR,angij,di,auvwi,errd,erra, elttwin]) # ICI ICI
                                listpairplan.append([angij,uvwi,planiR,uvwj,planjR]) # cest un permier tri sommaire
##            print
##            for el in listweakplanesAxes0: print el
            listweakplanesAxes1 = [] # second tri plus pousse
            listtokeep=[]
            compt = 0
            ComptMax = len(listweakplanesAxes0)
            for c in listweakplanesAxes0:
                perc = (100*compt) / ComptMax
                text = str(" " + str(perc) + " %")
                EtatRun.set(text)
                fcalc.update()
                axe,v1,plan1R,v2,plan2R = c[:5]
                sg,w1,w2 = c[-1][:3]
                if [axe,v1,w1.tolist(),plan1R] not in listtokeep:
                    listweakplanesAxes1.append(c)
                    for g in CRYST.sym():
                        gplan1 = dot(transpose(g),array(plan1R)).tolist()
                        gaxe = dot(g,array(axe)).tolist()
                        gv1 = dot(g,array(v1)).tolist()
                        gw1 = dot(g,array(w1)).tolist()
                        listtokeep.append([gaxe,gv1,gw1,gplan1])

##            listweakplanesAxes2 = [] # troisieme tri avec les matrices de distortion
##            listF=[]
##            for c in listweakplanesAxes1:
##                distF = c[-1][-1] #c[-1] = info, et le dernier elt est F
##                flag= 1
##                for Fi in listF:
##                    if allclose(distF,Fi):
##                        flag = 0
##                        break
##                if flag:
##                    listweakplanesAxes2.append(c)
##                    for g in CRYST.sym():
##                        listF.append(dot(dot(g,distF),inv(g)))

            listweakplanesAxes2 = [] # troisieme tri avec les matrices de correspondence = Pb : dtow different weak planes can have the same correspodence matrix (Pericline in feldspar)
            listPandC = []
            compt = 0
            ComptMax = len(listweakplanesAxes1)
            for c in listweakplanesAxes1:
                axe,v1,plan1R,v2,plan2R = c[:5]
                perc = (100*compt) / ComptMax
                text = str(" " + str(perc) + " %")
                EtatRun.set(text)
                fcalc.update()
                matC = c[-1][-2] #c[-1] = info, et le dernier elt est F
                flag= 1
                for PandC in listPandC: 
                    plan1R2,plan2R2 = PandC[:2]
                    listC = PandC [-1]
                    if (plan1R2 ==plan1R or plan1R2 == plan2R):
                        for Ci in listC:
                            if my_allclose(matC,Ci): # A VOIR ICI 7 sept
                                flag = 0
                                break
                if flag:
                    listweakplanesAxes2.append(c)
                    listC=[]
                    for g in CRYST.sym():
                        listC.append(dot(dot(g,matC),inv(g)))
                    listPandC.append([plan1R,plan2R,listC])
        
            if ModeWeak==2:
                EtatRun.set("")
                fcalc.update()
            
            return listweakplanesAxes2

        def do_calc_Axeweak():
            global listweakplanesAxes
            if axePi!=[0,0,0]: listweakplanesAxes = do_calc_AxeweakInfo(axePi)
            else : do_calc_AxeweakAll()

        def do_calc_AxeweakAll():
            global listweakplanesAxes, fcalc
            listweakplanesAxes = []
            excludelist = []
            lplus = arange(0,uvwmax+1,1)
            lmoins = arange(-1,-uvwmax-1,-1) 
            lind = concatenate((lplus,lmoins))
            listaxes = []
            for u in lind:
                for v in lind:
                    for w in lind:
                        direct = [u,v,w]
                        if direct == [0,0,0] or (direct in excludelist): pass
                        else:
                            listaxes.append(direct)
                            adirect = array(direct)
                            for g in CRYST.sym():
                                newdirect = dot(g,adirect)
                                excludelist.append(newdirect.tolist())
            compt = 0
            ComptMax = len(listaxes)
            for axis in listaxes:
                perc = (100*compt) / ComptMax
                text = str(" " + str(perc) + " %")
                EtatRun.set(text)
                fcalc.update()
                compt+=1
                listweakplanesAxes0 = do_calc_AxeweakInfo(axis)
                if listweakplanesAxes0:
                    listweakplanesAxes.extend(listweakplanesAxes0)
            EtatRun.set("")
            fcalc.update()

        def showtwininfo():

            def rationalize2(x):
                p,q= rationalize(x)
                res=1
                if abs(p)<0.00001: res = "0"
                else:
                    if p*q>0:
                        if abs(q)==1: res = str(abs(p))
                        else: res= str(abs(p))+"/"+str(abs(q))
                    else:
                        if abs(q)==1: res = "-"+str(abs(p))
                        else: res= "-"+str(abs(p))+"/"+str(abs(q))
                return res
            def writep(p):
                return "("+str(p[0])+","+str(p[1])+","+str(p[2])+")"

            ffInfoTwin = open("ListInfoTwin.txt", 'w')   
            axe,uvwi,planiR,uvwj,planjR,angij,di,auvwi,errd,erra, infotwin = listtoshow[twin]
            if infotwin: sg,OAi,OPj,matorient,CorMat,DistMat = infotwin
            else: sg,OAi,OPj,matorient,CorMat,DistMat = 0,[0,0,0],[0,0,0],Identity,Identity,Identity
            sgverif = ShearMyFormula(DistMat,CRYST)
            #print "Mean Strain (My Formula) = ", sgverif
            sgverifB = ShearMyFormulaB(DistMat,CRYST)
            #print "Mean Strain (My Formula B) = ", sgverifB
            sshear = ShearBevisCrocker(CorMat,CRYST)
            #print "sg Bevis Crocker = ", sshear
            TwinBox.delete(0,END)
            text= "Generalized strain of F (gs) = " + str(round(sg,5))
            TwinBox.insert(END, text)
            ffInfoTwin.write(text+"\n")
            text= "Shear deduced of C by Bevis-Crocker's formula = " + str(round(sshear,5))
            TwinBox.insert(END, text)
            ffInfoTwin.write(text+"\n"+"\n")
            text= "p1 = " + writep(planiR)+ ", p2 = " + writep(planjR)
            TwinBox.insert(END, text)
            ffInfoTwin.write(text+"\n"+"\n")
            TwinBox.insert(END, "")
            text= "U1 = U2 = " + str(axe)
            TwinBox.insert(END, text)
            ffInfoTwin.write(text+"\n")
            text= "V1 = " + str(uvwi)+ ", V2 = " + str(uvwj)
            TwinBox.insert(END, text+"\n")
            ffInfoTwin.write(text+"\n")
            text= "W1 = "+str(OAi)+ ",  W2 = "+str(OPj)
            TwinBox.insert(END, text)
            ffInfoTwin.write(text+"\n")
            gq = det(array([axe,uvwi,OAi]))
            text= "gq = "+ str(abs(gq))
            TwinBox.insert(END, text+"\n"+"\n")
            TwinBox.insert(END, " ")
            
            nbl = 8
            # correspondence matrix
            #======================
            text = "Correspondance Matrix C (twin => parent') "
            TwinBox.insert(END, text)
            ffInfoTwin.write(text+"\n")
            text = tabify("  ",nbl) + tabify(rationalize2(CorMat[0,0]))  + tabify(rationalize2(CorMat[0,1]))  + tabify(rationalize2(CorMat[0,2]))
            TwinBox.insert(END, text)
            ffInfoTwin.write(text+"\n")
            text = tabify("  ",nbl)  + tabify(rationalize2(CorMat[1,0])) + tabify(rationalize2(CorMat[1,1]))  + tabify(rationalize2(CorMat[1,2]))
            TwinBox.insert(END, text)
            ffInfoTwin.write(text+"\n")
            text = tabify("  ",nbl) + tabify(rationalize2(CorMat[2,0])) +  tabify(rationalize2(CorMat[2,1])) +  tabify(rationalize2(CorMat[2,2]))
            TwinBox.insert(END, text)
            ffInfoTwin.write(text+"\n"+"\n")
            TwinBox.insert(END, " ")
            TwinBox.insert(END, " ")               
            # distortion matrix
            #======================
            text = "Distortion Matrix F (parent => parent') "
            TwinBox.insert(END, text)
            ffInfoTwin.write(text+"\n")
            text = tabify("  ",nbl) + tabify(cut(DistMat[0,0]))  + tabify(cut(DistMat[0,1]))  + tabify(cut(DistMat[0,2]))
            TwinBox.insert(END, text)
            ffInfoTwin.write(text+"\n")
            text = tabify("  ",nbl)  + tabify(cut(DistMat[1,0])) + tabify(cut(DistMat[1,1]))  + tabify(cut(DistMat[1,2]))
            TwinBox.insert(END, text)
            ffInfoTwin.write(text+"\n")
            text = tabify("  ",nbl) + tabify(cut(DistMat[2,0])) +  tabify(cut(DistMat[2,1])) +  tabify(cut(DistMat[2,2]))
            TwinBox.insert(END, text+"\n"+"\n")
            ffInfoTwin.write(text)
            TwinBox.insert(END, " ")
            ffInfoTwin.close()

            # orientation matrix
            #======================
            showorient(matorient)
            TwinBox.update()
                        
 
        def showTwinWeakProp0(event):
            global twinold,twin
            twinold = twin
            try:
                twin = [int(x) for x in Listd.curselection()][0]
            except: pass
            showtwininfo()

        def showTwinWeakPropDown(event):
            global twinold,twin
            twinold = twin
            if twin<len(listtoshow)-1:twin+= 1
            showtwininfo()
            
        def showTwinWeakPropUp(event):
            global twinold,twin
            twinold = twin
            if twin>0:twin-= 1
            showtwininfo()
            

        def GiveInfo(event):
            Pi = '\u03C0'
            NearEq = '\u2248'
            texti = "A weak plane is a plane (hkl)1 that can be transformed into (hkl)1 by a small distortion while maintaining one direction invariant\n"
            texti = texti + "It is (hkl)1 = (U1,V1) and (hkl)2 = (U2,V2) such that U1 = U2, and V1 "+NearEq + " V2, i.e close norms, and angle(U1,V1) "+NearEq+ "angle(U2,V2) \n"
            texti = texti + "if (hkl)1 = -(hkl)2 the plane is called "+Pi+"-plane. Please note that this plane is only globally invariant, but not fully invariant\n "
            texti = texti + "The axial twins are based on the weak planes and a third direction W1 that should be close to W2 \n"
            texti = texti + "Two crystals form an axial twin if U1 = U2, det(U1,V1,W1) = det(U2,V2,W2) without being zero. \n"
            texti = texti + "The notion of shear amplitude s should be replaced by the more generalized concept of transformation strain (gs)."
            tkinter.messagebox.showinfo('================== Information ===================', texti)

        def ExportList(event): # en fait les fichiers on deja ete crees en meme temps que les donnees ont ete affichees dans les listbox
            os.startfile(dirtwin)

        def ExportTwinInfo(event):
            os.startfile(dirtwin)

        def build_fcalc():
            global fcalc, entry_uvwmax, entry_distmax, entry_axePi, entry_errdistmax, entry_errangmax, entry_sgmax,Listd, TwinBox, twin, twinold
            global listtoshow, ftwinInfo
            ffWeak = open("CurrentListWeakPlanes.txt", 'w')
            twinold = twin
            esp = '        '
            esp2 = '  '
            deg_u = '\u00ba'
            Angst_u = '\u212B'
            Pi_u = '\u03C0'
            Delta_u = '\u0394'
            NearEq_u = '\u2248'
            Arrow_u = '\u2192'
            deg = deg_u.encode('utf-8')
            Angst = Angst_u.encode('utf-8')
            Pi = Pi_u.encode('utf-8')
            Delta = Delta_u.encode('utf-8')
            NearEq = NearEq_u.encode('utf-8')
            Arrow = Arrow_u.encode('utf-8')
            try: fcalc.destroy()
            except: pass
            fcalc = Frame(frame)
            fmode = Frame(fcalc)
            if ModeWeak ==1: col1,col2,col3 = "blue","black","black"
            elif ModeWeak ==2: col1,col2,col3 = "black","blue","black"
            else: col1,col2,col3 = "black","black","blue"
            buttonModeWeak = Button(fmode, foreground=col1, background="grey")
            buttonModeWeak ["text"] = "Weak "+Pi + " Planes "
            buttonModeWeak.bind("<Button>", WeakMode1)
            buttonModeWeak.grid(row=0, column=1,  sticky = E ,ipadx=20)
            buttonModeWeak2 = Button(fmode, foreground=col2, background="grey")
            buttonModeWeak2 ["text"] = "Weak Planes of Axial Twins "
            buttonModeWeak2.bind("<Button>", WeakMode2)
            buttonModeWeak2.grid(row=0, column=2,  sticky = E ,ipadx=20)
            buttonModeWeak3 = Button(fmode, foreground=col3, background="grey")
            buttonModeWeak3 ["text"] = Arrow + " Transformation Matrices"
            buttonModeWeak3.bind("<Button>", WeakMode3)
            buttonModeWeak3.grid(row=0, column=3,  sticky = E ,ipadx=20)
            LabelRun = Label(fmode, textvariable=EtatRun, width = 20)
            LabelRun.grid (row=0, column=4, ipadx=20, sticky='W')
            buttonModeInfo = Button(fmode, foreground="white", background="grey")
            buttonModeInfo ["text"] = "Info"
            buttonModeInfo.bind("<Button>", GiveInfo)
            buttonModeInfo.grid(row=0, column=5,  sticky = E ,ipadx=20)
            fmode.grid(row=0, column=1,columnspan=8)
            fflag = Frame(fcalc)
            if ModeWeak==1:
                l = Label(fflag, text="______ List of  weak " +Pi+"-planes  _____")
                l.grid (row=2, column=3, columnspan=8)
            elif ModeWeak==2:
                lt = Label(fflag, foreground="black", text="Index max for U1 = U2 ").grid(row=2, column=4, sticky=W + E)
                entry_uvwmax = Entry(fflag, width=4)
                f = str(uvwmax)
                f = f[:f.find(".") + 2]
                entry_uvwmax.insert(0, f)
                entry_uvwmax.bind("<Return>", attr_par)
                entry_uvwmax.grid(row=2, column=5, sticky=E + W)
            if ModeWeak<=2:
                lt = Label(fflag, foreground="black", text="Reticular distance max " + "("+Angst +") = ").grid(row=3, column=4, sticky=W + E)
                entry_distmax = Entry(fflag, width=4)
                f = str(distmax)
                f = f[:f.find(".") + 2]
                entry_distmax.insert(0, f)
                entry_distmax.bind("<Return>", attr_par)
                entry_distmax.grid(row=3, column=5, sticky=E + W)
                lt = Label(fflag, foreground="black", text="Tolerance on distance (%)= ").grid(row=4, column=4, sticky=W + E)
                entry_errdistmax = Entry(fflag, width=4)
                entry_errdistmax.insert(0,errdistmax)
                entry_errdistmax.bind("<Return>", attr_par)
                entry_errdistmax.grid(row=4, column=5, sticky=E + W)
                lt = Label(fflag, foreground="black", text="Tolerance on angle "+"("+deg+") = ").grid(row=5, column=4, sticky=W + E)
                entry_errangmax = Entry(fflag, width=4)
                entry_errangmax.insert(0,errangmax)
                entry_errangmax.bind("<Return>", attr_par)
                entry_errangmax.grid(row=5, column=5, sticky=E + W)
                lt = Label(fflag, foreground="black")
                #if ModeWeak==1: pass #lt["text"] = "Near "+Pi+"-rotation axis (filter option)"
                if ModeWeak==1 or ModeWeak==2:
                    lt["text"] = "Axis U1 = U2 (filter option)"
                    lt["height"] = 1
                    lt.grid(row=6, column=4, sticky=E + W)
                    entry_axePi= [None for i in range(3)]
                    for i in range(3):
                        entry_axePi[i] = Entry(fflag, width=7)
                        entry_axePi[i].insert(0, axePi[i])
                        entry_axePi[i].bind("<Return>", attr_par)
                        entry_axePi[i].grid(row=6, column=i + 5, sticky=E + W)
                    Label(fflag, foreground="black", text = "(write 0,0,0 to ignore the filter)").grid(row=6, column=8, sticky=E + W)
                if ModeWeak==2:
                    lt = Label(fflag, foreground="black", text="Generalized Strain max = ").grid(row=7, column=4, sticky=W + E)
                    entry_sgmax = Entry(fflag, width=4)
                    entry_sgmax.insert(0,sgmax)
                    entry_sgmax.bind("<Return>", attr_par)
                    entry_sgmax.grid(row=7, column=5, sticky=E + W)
                    Label(fflag, foreground="black", text = "(to anticipate the twin)").grid(row=7, column=6, columnspan=3, sticky=E + W)
            
            if rankweak == 0:
                bgdist,bgang = "blue", "black"
            elif rankweak == 1:
                bgdist,bgang = "black", "blue"
            if rankweak == 0 or rankweak == 1: # only for the weak planes (pi or not)
                buttonRankdist = Button(fflag, foreground=bgdist, background="light grey")
                buttonRankdist ["text"] = "Sort with "+Delta+"d/d(%)"
                buttonRankdist .bind("<Button>", rank_errdist)
                buttonRankdist.grid(row=8, column=2, sticky = E ,ipadx=20)
                buttonRankang = Button(fflag, foreground=bgang, background="light grey")
                buttonRankang ["text"] = "Sort with "+Delta+"ang. ("+deg+")"
                buttonRankang .bind("<Button>", rank_errang)
                buttonRankang.grid(row=8, column=3, sticky = E ,ipadx=20)
            if rankweak == 3:
                bgdist,bgang = "blue", "black"
            elif rankweak == 4:
                bgdist,bgang = "black", "blue"
            if rankweak == 3 or rankweak == 4: # only for the weak twins
                Label(fflag, foreground="black", text = "").grid(row=7, column=3, sticky=E + W)
                buttonRankp = Button(fflag, foreground = bgdist, background="light grey")
                buttonRankp ["text"] = "Sort with axis"
                buttonRankp .bind("<Button>", rank_errp)
                buttonRankp.grid(row=8, column=2, sticky = E ,ipadx=20)
                buttonRanks = Button(fflag, foreground = bgang, background="light grey")
                buttonRanks ["text"] = "Sort with gs values"
                buttonRanks .bind("<Button>", rank_errs)
                buttonRanks.grid(row=8, column=3, sticky = E ,ipadx=20)  
            fflag.grid (row=2, column=2, columnspan=6, sticky = "W")
            if ModeWeak==1 or ModeWeak==2:
                frame2 = Frame(fcalc, width=100, borderwidth=4)
                l = Label(frame2, text="Current List of Weak planes", background="grey").grid (row=1, column=2, columnspan=2,sticky = W)
                Listd = Listbox(frame2, font = ('Helvetica', '9'),width=180, height=40)
            else:
                frame2 = Frame(fcalc, width=100, borderwidth=4)
                l = Label(frame2, text="Current List of Weak planes", background="grey").grid (row=1, column=2,sticky = W )
                Listd = Listbox(frame2, font = ('Helvetica', '9'), width=120, height=40)
                Listd.bind("<ButtonRelease>", showTwinWeakProp0)
                Listd.bind("<Up>", showTwinWeakPropUp)
                Listd.bind("<Down>", showTwinWeakPropDown)
            if ModeWeak==1 or ModeWeak==2 or ModeWeak==3:
                buttonModeExportList = Button(frame2, foreground="dark green", background="grey")
                buttonModeExportList ["text"] = "Export Current List of Twins"
                buttonModeExportList.bind("<Button>", ExportList)
                buttonModeExportList.grid(row=1, column=2,  sticky = E ,ipadx=20)
                
            Listd.grid (row=2, column=2)
            s = Scrollbar(frame2)
            s.grid(row=2, column=1, sticky="NS")
            s.config(command=Listd.yview)
            Listd.config(yscrollcommand=s.set)
##            flagfilter = 0
##            for a in axePi:
##                if a: flagfilter=1
##                listaxePi = []
##                for g in CRYST.sym():listaxePi.append(dot(g,array(axePi)).tolist())             

            def take4(elem):
                return elem[4]
            def take5(elem):
                return elem[5]
            def take6(elem):
                return elem[6]
            def take7(elem):
                return elem[7]
            def take8(elem):
                return elem[8]
            def take9(elem):
                return elem[9]
            def takesg(elem):
                return elem[-1][0]

            if ModeWeak ==1:
                listtoshow = copy.deepcopy(listweakplanes)
                #print "rankweak", rankweak
                if rankweak ==0: listtoshow.sort(key = take4)
                elif rankweak ==1: listtoshow.sort(key = take6)
                for weakplane in listtoshow:
                    uvwi,uvwj,axe180,di,errd,angi,erra,planij = weakplane
                    #if not flagfilter or (flagfilter and axe180 in listaxePi):
                    if 1:
                        #print uvwi,uvwj,axe180,di,angi,planij
                        sdi = str(round(di,5))[:5]
                        while len(sdi) <= 5: sdi = sdi + '0'
                        sangi = str(round(angi,5))[:5]
                        while len(sangi) <= 5: sangi = sangi + '0'
                        ed = str(round(100*errd,5))[:5]
                        while len(ed) <= 5: ed = ed + '0'
                        ad = str(round(erra,5))[:5]
                        while len(ad) <= 5: ad = ad + '0'
                        textplan = " Weak plane = " + "(" + str(planij[0]) + esp2 + str(planij[1]) + esp2 + str(planij[2]) + ")."+"\t"
                        textaxis = ". V1 = " + str(uvwi)+ NearEq +" V2 = "+str(uvwj)+ ".  Their length "+NearEq+" " + sdi+ "  " + Angst + esp2 +" Their angle with " +Pi+"-Axis = "+ sangi + deg+"\t"
                        texterr = ".   "+Delta+"d/d(%) = "+ed + "  "+Delta+"ang. ("+deg+") = " + ad +"\n"
                        text = textplan + " U1 = U2 = " + str(axe180) + esp2 + textaxis + texterr
                        Listd.insert(END, text)
                        ffWeak.write(text)
            elif ModeWeak ==2:
                if len(listweakplanesAxes)==0: Listd.insert(END, "Check that the zone axis of the weak planes is not 0,0,0, or increase the max reticular distance, or increase the tolerances")
                listtoshow = copy.deepcopy(listweakplanesAxes)
                if rankweak ==0: listtoshow.sort(key = take8)
                elif rankweak ==1: listtoshow.sort(key = take9)
                for weakplane in listtoshow:
                    axe,uvwi,plani,uvwj,planj,angij,di,angi,errd,erra, infotwin = weakplane
                    try : sg = infotwin[0]
                    except: sg = 0.
                    #angij = min(angij, abs(180-angij),abs(180+angij))
                    if 1: # to refine if one needs to filter according to the rotation angij 90, 120,180
                        sdi = str(round(di,5))[:5]
                        while len(sdi) <= 5: sdi = sdi + '0'
                        sangi = str(round(angi,5))[:5]
                        while len(sangi) <= 5: sangi = sangi + '0'
                        sangij = str(round(angij,5))[:5]
                        while len(sangij) <= 5: sangij = sangij + '0'
                        ed = str(round(100*errd,5))[:5]
                        while len(ed) <= 5: ed = ed + '0'
                        ad = str(round(erra,5))[:5]
                        while len(ad) <= 5: ad = ad + '0'
                        textplani = " U1 = U2 = " + str(axe)+". Weak plane:  p1 = (" + str(plani[0]) + esp2 + str(plani[1]) + esp2 + str(plani[2]) + ") "
                        textplanj = NearEq + " p2 = (" + str(planj[0]) + esp2 + str(planj[1]) + esp2 + str(planj[2]) + ")."+" Angle (p1,p2) = "+ sangij +deg
                        textaxis = ". V1 = " + str(uvwi)+ NearEq + " V2 = "+str(uvwj)+ ". Their length "+NearEq+" " + sdi+ "  " + Angst + esp2 +" and (V1,U1) "+NearEq+" (V2,U2) "+NearEq+ sangi + deg
                        texterr = ". "+Delta+"d/d(%) = "+ed + "  "+Delta+"ang. ("+deg+") = " + ad +"\t"
                        text = textplani + textplanj + esp2 + textaxis + texterr + ". gs = " + str(round(sg,5)) +"\n"
                        Listd.insert(END, text)
                        ffWeak.write(text)
            elif ModeWeak ==3 or ModeWeak ==4:
                listtoshow = copy.deepcopy(listweakplanesAxes)
                if rankweak ==3: pass
                elif rankweak ==4: listtoshow.sort(key = takesg)
                for weakplane in listtoshow:
                    axe,uvwi,plani,uvwj,planj,angij,di,angi,errd,erra, infotwin = weakplane
                    try : sg = infotwin[0]
                    except: sg = 0.
                    angij = min(angij, abs(180-angij),abs(180+angij))
                    if 1: # to refine if one needs to filter according to the rotation angij 90, 120,180
                        sdi = str(round(di,5))[:5]
                        while len(sdi) <= 5: sdi = sdi + '0'
                        sangi = str(round(angi,5))[:5]
                        while len(sangi) <= 5: sangi = sangi + '0'
                        sangij = str(round(angij,5))[:5]
                        while len(sangij) <= 5: sangij = sangij + '0'
                        ed = str(round(100*errd,5))[:5]
                        while len(ed) <= 5: ed = ed + '0'
                        ad = str(round(erra,5))[:5]
                        while len(ad) <= 5: ad = ad + '0'
                        textplani = " U1 = U2 = " + str(axe)+". Weak plane:  p1 = (" + str(plani[0]) + esp2 + str(plani[1]) + esp2 + str(plani[2]) + ") "
                        textplanj = NearEq + " p2 = (" + str(planj[0]) + esp2 + str(planj[1]) + esp2 + str(planj[2]) + ")."
                        textaxis = " V1 = " + str(uvwi)+ NearEq+" V2 = "+str(uvwj)
                        texterr = ".   "+Delta+"d/d(%) = "+ed + "  "+Delta+"ang. ("+deg+") = " + ad 
                        text = textplani + textplanj + esp2 + textaxis + texterr + ". gs = " + str(round(sg,5)) +"\n"
                        Listd.insert(END, text)
                        ffWeak.write(text)
                    
            frame2.grid (row=6, column=1,columnspan=4)
            
            if (ModeWeak ==3 or ModeWeak ==4):
                ftwinInfo = Frame(fcalc, width=150, borderwidth=4)
                l = Label(ftwinInfo, text="Information on the Selected Twin ", background="grey").grid (row=1, column=2,sticky = W)
                buttonModeExportInfo = Button(ftwinInfo, foreground="dark green", background="grey")
                buttonModeExportInfo ["text"] = "Export Info of the Selected Twin"
                buttonModeExportInfo.bind("<Button>", ExportTwinInfo)
                buttonModeExportInfo.grid(row=1, column=3,  sticky = E)
                TwinBox = Listbox(ftwinInfo,font = ('Courier', '10'), width=80, height=40)
                TwinBox.grid (row=2, column=2, columnspan=2)
                sc = Scrollbar(ftwinInfo)
                sc.grid(row=2, column=1, sticky="NS")
                sc.config(command=TwinBox.yview)
                TwinBox.config(yscrollcommand=sc.set)
                ftwinInfo.grid(row=6, column=6, sticky=N)
                #if twin!=-1: Listd.activate(twin)
                #showtwininfo()
            
            ffWeak.close()
            fcalc["borderwidth"] = 1
            fcalc["relief"] = SOLID
            fcalc.grid(row=1, column=2, sticky=N)
        
        runtwin()
    #===============================================================================================================
    # ====================================== END OF WEAK TWINS =====================================================
    #===============================================================================================================

    
    def calc_listdhkl():
        global fcalc, entry_freqmax, entry_dmin, freqmax, SymFlag

        SymFlag = 1

        try: fcalc.destroy()
        except:pass

        def attr_freqmax(event):
            global fcalc, freqmax
            freqmax = string.atof(entry_freqmax.get())
            fcalc.destroy()
            build_fcalc()

        def attr_dmin(event):
            global fcalc, freqmax
            dmin = string.atof(entry_dmin.get())
            freqmax = 1. / dmin
            fcalc.destroy()
            build_fcalc()

        def attr_sym():
            global fcalc, SymFlag
            SymFlag = (SymFlag + 1) % 2
            fcalc.destroy()
            build_fcalc()

        def build_fcalc():
            global fcalc, entry_freqmax, entry_dmin, SymFlag

            esp = '        '
            esp2 = '  '
            Angst = '\u212B'
            try: fcalc.destroy()
            except: pass
            fcalc = Frame(frame)
            l = Label(fcalc, text="______ List of interplanar distances _____")
            l.grid (row=0, column=1, columnspan=4)
            fflag = Frame(fcalc)
            Checkbutton (fflag, text="With/without symmetries", variable=tkSym, command=attr_sym).grid (row=2, column=1, columnspan=6)
            #Radiobutton (fflag, text="Without", variable=tkSym, value=(SymFlag+1)%2,command = attr_sym).grid (row=2, column=1,sticky="W")
            lt = Label(fflag, foreground="black", text="dhkl min = ").grid(row=3, column=1, sticky=W + E)
            entry_dmin = Entry(fflag, width=4)
            entry_dmin.insert(0, 1. / freqmax)
            entry_dmin.bind("<Return>", attr_dmin)
            entry_dmin.grid(row=3, column=2, sticky=E + W)
            lt = Label(fflag, foreground="black", text=Angst, height=1).grid(row=3, column=3, sticky=W)
            lt = Label(fflag, foreground="black", text=" freq. max = ").grid(row=3, column=4, sticky=W + E)
            entry_freqmax = Entry(fflag, width=4)
            f = str(freqmax)
            f = f[:f.find(".") + 2]
            entry_freqmax.insert(0, f)
            entry_freqmax.bind("<Return>", attr_freqmax)
            entry_freqmax.grid(row=3, column=5, sticky=E + W)
            lt = Label(fflag, foreground="black", text="1/"+Angst, height=1).grid(row=3, column=6, sticky=W)
            fflag.grid (row=1, column=1, columnspan=4)
            lt = Label(fcalc, foreground="black", text="", height=1).grid(row=4, column=1, columnspan=2, sticky=W)
            frame2 = Frame(fcalc, width=50, borderwidth=4)
            lt = Label(frame2, foreground="black", text="  dhkl (Ang.)" + esp + "h  k l" + esp + "  Intensity", height=1).grid(row=1, column=1, columnspan=2, sticky=W)
            Listd = Listbox(frame2, width=35, height=30)
            Listd.grid (row=2, column=2)
            s = Scrollbar(frame2)
            s.grid(row=2, column=1, sticky="NS")
            s.config(command=Listd.yview)
            Listd.config(yscrollcommand=s.set)
            for hkl in CRYST.listdhkl(SymFlag, freqmax):
                d = str(hkl[0])[:5]
                while len(d) <= 5:
                    d = d + '0'
                text = str(d + esp + esp2 + str(hkl[1]) + esp2 + str(hkl[2]) + esp2 + str(hkl[3]) + esp + str(round(hkl[4], 4)))
                #print text
                Listd.insert(END, text)
                text = "---------------------------------------------------------------------------"
                Listd.insert(END, text)
            frame2.grid (row=5, column=1, columnspan=4)
            fcalc["borderwidth"] = 1
            fcalc["relief"] = SOLID
            fcalc.grid(row=1, column=2, sticky=N)

        build_fcalc()


    def calc_listrot():
        global fcalc, axerot, angrot,SymFlag

        try: fcalc.destroy()
        except:pass

        SymFlag = 1

        tkSym = IntVar()
        arot = DoubleVar()
        arot.set(0.0)
        urot = IntVar()
        urot.set(0)
        vrot = IntVar()
        vrot.set(0)
        wrot = IntVar()
        wrot.set(1)

        def attr_sym():
            global fcalc, SymFlag
            SymFlag = (SymFlag + 1) % 2
            fcalc.destroy()
            build_fcalc()
            showlist(1)

        def showlist(event):
            frame2 = Frame(fcalc, width=50, borderwidth=4)
            Listrot = Listbox(frame2, width=35, height=30)
            Listrot.grid (row=2, column=2)
            s = Scrollbar(frame2)
            s.grid(row=2, column=1, sticky="NS")
            s.config(command=Listrot.yview)
            Listrot.config(yscrollcommand=s.set)
            axerot = array([urot.get(),vrot.get(),wrot.get()])
            angrot = arot.get()
            TS = CRYST.structTensDir
            mrotortho = calculeMatRot(angrot,dot(TS,axerot)) 
            mrot = dot(inv(TS),dot(mrotortho,TS))
            for roti in ListEquivRot(mrot, CRYST.sym(),TS,1,SymFlag): # 1 to get the matrix in cryst frame, 0 to get in ortho frame
                ang,axis,mk = roti
                esp = ' ' 
                if len(axis)>0:
                    try: text = str(cut(ang) + " Deg.   " + "\t\t" + " Axis = " + esp + str(axis[0]) + esp + str(axis[1]) + esp + str(axis[2]))
                    except: text = ''
                else: text = "Identity"
                #print text
                if text: Listrot.insert(END, text)
                text = "---------------------------------------------------------------------------"
                Listrot.insert(END, text)
            frame2.grid (row=2, column=1, columnspan=6)

        def build_fcalc():
            global fcalc, SymFlag
            esp = '        '
            esp2 = '  '
            try: fcalc.destroy()
            except: pass
            fcalc = Frame(frame)
            fflag = Frame(fcalc)
            lt = Label(fflag, foreground="black", text="Rotation Angle = ").grid(row=1, column=1, sticky=W + E)
            Labelarot = Entry(fflag, textvariable=arot,width=4)
            Labelarot.bind("<Return>", showlist)
            Labelarot.grid (row=1, column=2, ipadx=1, sticky='EW')
            lt = Label(fflag, foreground="black", text=" Deg.").grid(row=1, column=3, sticky=W + E)
            lt = Label(fflag, foreground="black", text="Axis = ", height=1).grid(row=1, column=4, sticky=W)
            Labelurot = Entry(fflag, textvariable=urot,width=4)
            Labelurot.bind("<Return>", showlist)
            Labelurot.grid (row=1, column=5,  sticky='EW')
            Labelvrot = Entry(fflag, textvariable=vrot,width=4)
            Labelvrot.bind("<Return>", showlist)
            Labelvrot.grid (row=1, column=6,  sticky='EW')
            Labelwrot = Entry(fflag, textvariable=wrot,width=4)
            Labelwrot.bind("<Return>", showlist)
            Labelwrot.grid (row=1, column=7,  sticky='EW')
            Checkbutton (fflag, text="With/without symmetries", variable=tkSym, command=attr_sym).grid (row=1, column=8, columnspan=6)
            l = Label(fflag, text="______ List of equivalent rotations _____")
            l.grid (row=2, column=1, columnspan=8)
            fflag.grid(row=1, column=1, sticky=N)
            
            fcalc["borderwidth"] = 1
            fcalc["relief"] = SOLID
            fcalc.grid(row=1, column=2, sticky=N)
            
        build_fcalc()
        
    #=====================================================================================================================
    # Twinning
    #=====================================================================================================================
    
    def calc_listtwins():
        global fcalc,  TwinBox, shear_max, obliq_max, index_max, RankFlag
        global fresult, ListResP, ListResS
        global listresultTotPlanesDir, listresultTotShearsDir
        global listresultTotPlanesRec, listresultTotShearsRec
        global twintype

        RankFlag = 1
        tkRank = IntVar()

        try: fcalc.destroy()
        except:pass
        hkl_max = IntVar()
        hkl_max.set(3)
        index_max = IntVar()
        index_max.set(3)

        def is_int(x):
            return x - round(x) == 0

        def is_halfint(x):
            return is_int(x+1./2)

        def is_sameodd(p): # the 3 coordinates are all odd, or all even
            d = 1./2
            ceven = is_int(p[0]*d) and is_int(p[1]*d) and is_int(p[2]*d)
            codd = is_int((p[0]-1)*d) and is_int((p[1]-1)*d) and is_int((p[2]-1)*d)
            return ceven or codd

        def is_FCC_Dir(u):
            return is_int(sum(u))

        def is_BCC_Dir(u):
            return is_sameodd([2*u[0],2*u[1],2*u[2]])

        def is_FCC_Rec(p):
            return is_sameodd(p)

        def is_BCC_Rec(p):
            return is_int(sum(p)*(1./2))  

        def findUV(U,V,struct): # directions U and V of smallest size that belong to the U,V plane
            # the coordinates of U1 and V1 can be only integer or half integer
            # en fait jaurais du noter zz2 et non oz2
            oz2 = array([0,0,0]) # will be used to adjust the Bezout solution OZ to be a FCC or BCC direction
            if (struct == "FCC")  : # the sum of the coordinates of the vector OZ, U and V should be integer
                if is_FCC_Dir(U) and is_FCC_Dir(V):
                    Ures,Vres = U,V
                elif is_FCC_Dir(U) and not is_FCC_Dir(V):
                    Ures,Vres = U,2*V
                    oz2 = V
                elif is_FCC_Dir(V) and not is_FCC_Dir(U):
                    Ures,Vres = 2*U,V
                    oz2 = U
                elif not is_FCC_Dir(U) and not is_FCC_Dir(V):
                    Ures,Vres = U-V, U+V
                    oz2 = U
            elif (struct == "BCC") : # all the coordinates are of same type: all integers or all half integer
                if is_BCC_Dir(U) and is_BCC_Dir(V):
                    Ures,Vres = U,V
                elif is_BCC_Dir(U) and not is_BCC_Dir(V):
                    Ures,Vres = U,2*V
                    oz2 = V
                elif is_BCC_Dir(V) and not is_BCC_Dir(U):
                    Ures,Vres = 2*U,V
                    oz2 = U
                elif not is_BCC_Dir(U) and not is_BCC_Dir(V):
                    Ures,Vres = U-V, U+V
                    oz2 = U
            return oz2,Ures,Vres

        def findHK(H,K,struct): # plane H and K of smallest size that belong to the H,K plane
            # the coordinates of U1 and V1 can be only integer or half integer
            oz2 = array([0,0,0]) # will be used to adjust the Bezout solution OZ to be a FCC or BCC direction
            if (struct == "FCC")  : # the sum of the coordinates of the vector OZ, U and V should be integer
                if is_FCC_Rec(H) and is_FCC_Rec(V):
                    Hres,Kres = H,K
                elif is_FCC_Rec(H) and not is_FCC_Rec(K):
                    Hres,Kres = H,2*K
                    oz2 = K
                elif is_FCC_Rec(K) and not is_FCC_Rec(H):
                    Hres,Kres = 2*H,K
                    oz2 = H
                elif not is_FCC_Rec(H) and not is_FCC_Rec(K):
                    Hres,Kres = H-K, H+K
                    oz2 = H
            elif (struct == "BCC") : # all the coordinates are of same type: all integers or all half integer
                if is_BCC_Rec(H) and is_BCC_Rec(K):
                    Hres,Kres = H,K
                elif is_BCC_Rec(H) and not is_BCC_Rec(K):
                    Hres,Kres = H,2*K
                    oz2 = K
                elif is_BCC_Rec(K) and not is_BCC_Rec(H):
                    Hres,Kres = 2*H,K
                    oz2 = H
                elif not is_BCC_Rec(H) and not is_BCC_Rec(K):
                    Hres,Kres = H-K, H+K
                    oz2 = H
            return oz2,Hres,Kres

        def attr_rank(nb):
            global RankFlag, ListResP, ListResS
            RankFlag = nb
            if RankFlag==1:
                try: ListResS.destroy()
                except: pass
                build_ListResP()

            else:
                try: ListResP.destroy()
                except: pass
                build_ListResS()
            
        def attr_rank1Dir(event):
            global twintype
            twintype = "Dir"
            attr_rank(1)
            
        def attr_rank2Dir(event):
            global twintype
            twintype = "Dir"
            attr_rank(0)

        def attr_rank1Rec(event):
            global twintype
            twintype = "Rec"
            attr_rank(1)
            
        def attr_rank2Rec(event):
            global twintype
            twintype = "Rec"
            attr_rank(0)

        def rank_dist(OZ,ZI,ZA,ZB,ZC,ZD,dhkl):
            OA = OZ + ZA
            OB = OZ + ZB
            OC = OZ + ZC
            OD = OZ + ZD
            AI = -ZA + ZI
            BI = -ZB + ZI
            CI = -ZC + ZI
            DI = -ZD + ZI
            if twintype == "Dir":
                dA = vectDir(CRYST,AI).length()
                dB = vectDir(CRYST,BI).length()
                dC = vectDir(CRYST,CI).length()
                dD = vectDir(CRYST,DI).length()
                SymGroup = CRYST.sym()
            else:
                dA = vectRec(CRYST,AI).length()
                dB = vectRec(CRYST,BI).length()
                dC = vectRec(CRYST,CI).length()
                dD = vectRec(CRYST,DI).length()
                SymGroup = CRYST.symrec()
            a = [[dA/dhkl,AI.tolist(),OA.tolist(),"A"],[dB/dhkl,BI.tolist(),OB.tolist(),"B"],
                 [dC/dhkl,CI.tolist(),OC.tolist(),"C"],[dD/dhkl,DI.tolist(),OD.tolist(),"D"]]
            a.sort()
            b = []        
            for ra in a:
                flag = 1
                for rb in b:
                    if abs(ra[0]-rb[0])<0.0000001:
                        for s in SymGroup :
                            if length(ra[1] - dot(s,array(rb[1])))<0.001: flag = 0
                if flag: b.append(ra)
            return b

        def dirplane(p):
            h,k,l = p
            if h!=0 and k!=0 and l!=0:
                u,v = [-k,h,0],[0,-l,k]
            elif h!=0 and k!=0:
                u,v = [-k,h,0],[0,0,1]
            elif h!=0 and l!=0:
                u,v = [-l,0,h],[0,1,0]
            elif k!=0 and l!=0:
                u,v = [0,-l,k],[1,0,0] 
            elif h!=0:
                u, v = [0,1,0],[0,0,1]
            elif k!=0:
                u, v = [1,0,0],[0,0,1]
            elif l!=0:
                u, v = [1,0,0],[0,1,0]
            return u,v
                    
        def issymplane(p,twintype = "Dir"):
            # teste si le plan est un plan mirroir dans le cas Dir, ie le plan est completement invariant, ie toutes les directions dans le plan sont invariantes
            # et dans le cas Rec, teste si la direction "p" est complement ivariantes, ie tous les plans contenant cette direction sont invariants
            if twintype == "Dir": SymGroup = CRYST.sym()
            else: SymGroup = CRYST.symrec()
            u,v = dirplane(p)
            res = 0
            if p == [0,0,0]:
                pass
            else:
                for g in SymGroup[1:]:# Identity excluded
                    if dot(g,array(u)).tolist() == u and dot(g,array(v)).tolist() == v:
                        res = 1
                        break
            return res
        
        def calclistDir(event):
            global twintype
            twintype = "Dir"
            calclist()

        def calclistRec(event):
            global twintype
            twintype = "Rec"
            calclist()
            
        def structRec(struct):
            if struct == "FCC" : structE = "BCC"
            elif struct == "BCC" : structE = "FCC"
            elif struct == "I" : structE = "F"
            elif struct == "F" : structE = "I"
            elif struct == "I" : structE = "F"
            else: structE = struct
            return structE
            
        def calclist():
            global listresultTotPlanesDir, listresultTotShearsDir
            global listresultTotPlanesRec, listresultTotShearsRec
            global twinold, twin
            # les notations sont pour le cas Dir ie type I twin, K1 = hkl rationnel et nu1 a calculer
            # les memes notations sont utilisees pour typeII nu1 rationnel et K1 a calculer par meme algorithme que pour cas Dir mais dans l espace Rec
            
            twinold = -1
            twin = -1
            hklmax = hkl_max.get()
            indmax = index_max.get()
            if twintype == "Dir":
                TS = CRYST.structTensDir
                GMD = CRYST.metricTensDir # metric tensor for dir space [REC=>DIR]
                GME = CRYST.metricTensRec # metric tensor for rec space [DIR=>REC]
                struct = CRYST.struct
                sym = CRYST.symrec()
            elif twintype == "Rec":
                TS = CRYST.structTensRec
                GMD = CRYST.metricTensRec # metric tensor for dir space [REC=>DIR]
                GME = CRYST.metricTensDir # metric tensor for rec space [DIR=>REC]
                struct = structRec(CRYST.struct)
                sym = CRYST.sym()
            
            listresultTotPlanes = [] # list ranked by planes hkl
            listresultTotShears = [] # same list but ranked by shear values
            listhkltested = [] # list of planes to exclude because = plane of symmetry or already tested
            excludesym = 1 # to exclude the sym elements of the crystal, here the mirror planes
##            lowhkl = -hklmax
##            if CRYST.syst == "Triclinic" or CRYST.syst == "Monoclinic": lowhkl = -hklmax
##            else: lowhkl =  0
##            if CRYST.syst == "Triclinic": lowl = -hklmax
##            else: lowl = 0
            listhkl = []
            #if twintype == "Dir":
            if 1:
                if (struct == "FCC" or struct == "BCC"): hklmax = 2*hklmax
                lplus = arange(0,hklmax+1,1)
                lmoins = arange(-1,-hklmax-1,-1) 
                lind = concatenate((lplus,lmoins))
                for h in lind:
                    for k in lind:
                        for l in lind:
                            p = [h,k,l]
                            if p == [0,0,0] or (issymplane(p) and excludesym): pass
                            elif (p in listhkltested):pass
                            else:
                                ap = array(p)
                                for g in sym:
                                    newp = dot(g,ap)
                                    listhkltested.append(newp.tolist())
                                gcd = gcd3D(h,k,l)
                                cond1 = (gcd==1) and struct != "FCC"  and struct != "BCC"
                                cond2 =  (gcd==1 or gcd==2) and ((struct == "FCC" and is_FCC_Rec(p)) or (struct == "BCC" and is_BCC_Rec(p)))
                                if cond1 or cond2: listhkl.append([h,k,l,gcd])
                                
            for p in listhkl:
                h,k,l,gcd = p
                ap = array(p[:-1])
                apD = dot(GME,ap)
                dhkl2 = (1./dot(ap,apD)) # dhkl
                OH1 = dhkl2*apD # vector normal to the plane hkl such that its norm = dhkl
                # a voir pour la partie Rec car les 1/2 ne collent pas avec les plans
                # a finir donc
                if (struct != "FCC" and struct != "BCC"): 
                    OZ1,U,V = Bezout3D(h,k,l) # OZ1 = [u,v,w] a solution of uh+kv+vw = 1, U and V two vectors such that U.(h,k,l)=0 and V.(h,k,l)=0
                elif gcd == 1:
                    OZ1,U,V = Bezout3D(h,k,l)
                    oz2,U,V = findUV(0.5*U,0.5*V,CRYST.struct)
                elif gcd == 2:
                    OZ1,U,V = Bezout3D(h/2,k/2,l/2)
                    oz2,U,V = findUV(0.5*U,0.5*V,CRYST.struct)     
                    OZ1 = 0.5*OZ1
                    if struct == "FCC" and not is_FCC_Dir(OZ1): OZ1 = OZ1 + oz2
                    if struct == "BCC" and not is_BCC_Dir(OZ1): OZ1 = OZ1 + oz2

                BasisUV = transpose(array([U,V]))   
                listresult= []
                listind_q = [i for i in range(1,indmax+1)]
                for splane in [1,-1]: # explore above the plane or below the plane
                    for q in listind_q:
                        dhkl = q*sqrt(dhkl2)
                        OH = splane*q*OH1
                        OZ = q*OZ1
                        ZH = -OZ+OH
                        ZH_inUV = dot(LeftInv(BasisUV),ZH)
                        nU,rU = ft_min(ZH_inUV[0])
                        nV,rV = ft_min(ZH_inUV[1])
                        ZA = (nU*U + nV*V) # A is the closest point from H by coordinate, not necessarily by distance
                        sU = sign(rU)
                        if sU ==0: sU = 1
                        sV = sign(rV)
                        if sV ==0: sV = 1
                        ZB = ZA + sU*U
                        ZC = ZA + sV*V
                        ZD = ZA + sU*U + sV*V
                        AH = -ZA + ZH
                        ZI = ZA + 2*AH # Im = image of A by central symmetry around H
                        OA = OZ+ZA
                        OI = OZ+ZI
                        rankH = rank_dist(OZ,ZH,ZA,ZB,ZC,ZD,dhkl)
                        rankI = rank_dist(OZ,ZI,ZA,ZB,ZC,ZD,dhkl)
                        sHmin, HminH, OHmin, Hmin = rankH[0]    # Hmin = A,B,C ou D le plus proche de H
                        sImin, IminI, OImin, Imin = rankI[0]    # Imin = A,B,C ou D le plus proche de I
                        FGmode = "Tilted"
                        if Hmin == Imin: FGmode = "Normal"
                        else:
                            for ir in range(1,len(rankI)):
                                sIminir, IminIir, OIminir, Iminir = rankI[ir]
                                if abs(sIminir-sImin)<0.0001 and Iminir == Hmin:
                                    FGmode = "Normal"
                                    print("te remettre a programmer cette solution I")
                            for ir in range(1,len(rankH)):
                                sHminir, HminHir, OHminir, Hminir = rankH[ir]
                                if abs(sHminir-sHmin)<0.0001 and Hminir == Imin:
                                    FGmode = "Normal"
                                    print("te remettre a programmer cette solution H")

                        listresult.append([[h,k,l],q,U,V,OA,OImin,rankI,FGmode,sHmin]) # K1 = hkl, nu1 = transl, nu2 = OA, sHmin used for the calculation of the normal obliquity (Friedel)
                        for resi in rankI:
                            listresultTotShears.append([resi[0],[h,k,l],q,U,V,OA,OImin,resi[1:],FGmode,sHmin])
                listresultTotPlanes.append(listresult)

            listresultTotShears.sort()
            if twintype == "Dir":
                listresultTotPlanesDir = listresultTotPlanes
                listresultTotShearsDir = listresultTotShears
            else:
                listresultTotPlanesRec = listresultTotPlanes
                listresultTotShearsRec = listresultTotShears
            build_ListResP()

        def build_ListResP():
            global ListResP, ListResS
            ListResP = Listbox(fresult, selectmode=SINGLE, width=90, height=30)
            ListResP.grid (row=2, column=2)
            s = Scrollbar(fresult)
            s.grid(row=2, column=1, sticky="NS")
            s.config(command=ListResP.yview)
            ListResP.config(yscrollcommand=s.set)
            showlistP()     

        def build_ListResS():
            global ListResS, ListResP, twin,fresult,TwinBox

            ftwin = Frame(fcalc)
            l = Label(ftwin, text="Information on the selected twin (in the list of twins ranked by shear values) ", background="grey")
            l.grid (row=1, column=1, columnspan=2)
            TwinBox = Listbox(ftwin,font = ('Courier', '10'), width=90, height=30)
            TwinBox.grid (row=2, column=2)
            s = Scrollbar(ftwin)
            s.grid(row=2, column=1, sticky="NS")
            s.config(command=TwinBox.yview)
            TwinBox.config(yscrollcommand=s.set)
            ftwin.grid(row=2, column=2, sticky=N)

            ListResS = Listbox(fresult, font = ('Courier', '10'), selectmode=SINGLE, width=90, height=30)
            ListResS.grid (row=2, column=2)
            s = Scrollbar(fresult)
            s.grid(row=2, column=1, sticky="NS")
            s.config(command=ListResS.yview)
            ListResS.config(yscrollcommand=s.set)
            #fresult.grid(row=2, column=1, sticky=N)
            showlistS()
            ListResS.bind("<ButtonRelease>", showtwin0)
            ListResS.bind("<Up>", showtwinUp)
            ListResS.bind("<Down>", showtwinDown)


        def showlistP():
            global listResP, fresult, twin
            if twintype == "Dir": listresultTotPlanes = listresultTotPlanesDir
            else: listresultTotPlanes = listresultTotPlanesRec
            for resplane in listresultTotPlanes:
                ListResP.insert(END, "")
                text = "============================="
                ListResP.insert(END, text)
                p  = resplane[0][0]
                FGmode = resplane[0][-2]
                sHmin = resplane[0][-1]
                h,k,l = p
                OblN = arctan(sHmin)/ Degree
                if twintype == "Dir": text = " K1 = " + esp2 + str(h) + esp2 + str(k) + esp2 + str(l)  + esp2 + " Crystal Obliquity (Friedel)= " + cut(OblN) + '\u00ba'
                else: text = '\u03b7' + "1 = " + esp2 + str(h) + esp2 + str(k) + esp2 + str(l)  + esp2 + " Crystal Obliquity (Friedel)= " + cut(OblN) + '\u00ba'
                ListResP.insert(END, text) 
                for r in resplane:
                    ListResP.insert(END, "")
                    p,q,U,V, OA,OImin,list_rank,FGmode,sHmin = r
                    text = "q = " + str(q) + "  Mode = " + FGmode
                    ListResP.insert(END, text)
                    for shearinfo in list_rank:
                        shear, transl = shearinfo[0],shearinfo[1]
                        dirrat = IntegriseAxe(array(transl),0.001)
                        if twintype == "Dir": offset = angleVect(CRYST,vectDir(CRYST,transl),vectDir(CRYST,dirrat))
                        else: offset = angleVect(CRYST,vectRec(CRYST,transl),vectRec(CRYST,dirrat))
                        if abs(offset)<0.0001: offset = 0  # offset = ecart entre valeur exact et valeur rationnlise du vecteur
                        
                        if twintype =="Dir": Obl = angleVect(CRYST,vectDir(CRYST,OImin),vectDir(CRYST,array(OImin)+array(transl)))
                        else: Obl = angleVect(CRYST,vectRec(CRYST,OImin),vectRec(CRYST,array(OImin)+array(transl)))
                        if abs(Obl)<0.0001: Obl = 0 # obliquity = angle le vecteur avant et apres shear
                        
                        if twintype == "Dir": text = "Shear = " + str(cut(shear))+ " "+ '\u21d2' + " Obliquity = "+ cut(Obl) + '\u00ba' +".    " '\u03b7' + "1 = " + str(dirrat)
                        else: text = "Shear = " + str(cut(shear))+ " "+ '\u21d2' + " Obliquity = "+ cut(Obl) + '\u00ba' +".    "  + "K1 = " + str(dirrat)
                        if offset: text2 = '\u00b1' + cut(offset,5) + '\u00ba' 
                        else: text2 = ''
                        ListResP.insert(END, text+text2)

        def showlistS():
            global listResS, fresult, twin
            if twintype == "Dir": listresultTotShears = listresultTotShearsDir
            else: listresultTotShears = listresultTotShearsRec
            for r in listresultTotShears:
                shear,p,q,U,V,OA,OImin,list_s,FGmode,sHmin = r
                transl, OP, letterABCD = list_s 
                h,k,l = p
                dirrat = IntegriseAxe(array(transl),0.001)
                ur,vr,wr =dirrat
                if twintype == "Dir": offset = angleVect(CRYST,vectDir(CRYST,transl),vectDir(CRYST,dirrat))
                else:offset = angleVect(CRYST,vectRec(CRYST,transl),vectRec(CRYST,dirrat))
                if abs(offset)<0.0001: offset = 0
                if twintype == "Dir":
                    text = "Shear = " + str(cut(shear))+ "   K1 = (" + str(h) + esp2 + str(k) + esp2 + str(l) + ")"
                    if offset: text2 = "    " + '\u03b7' + "1 = " + str(dirrat) + '\u00b1' + cut(offset) + '\u00ba' 
                    else: text2 = "    " + '\u03b7' + "1 = [" +  str(ur) + esp2 + str(vr) + esp2 + str(wr) + "]"
                else:
                    text = "Shear = " + str(cut(shear))+ "   " + '\u03b7' + "1 = ["+ str(h) + esp2 + str(k) + esp2 + str(l) + "]"
                    if offset: text2 = "    " + "K1 = (" +  str(ur) + esp2 + str(vr) + esp2 + str(wr)+ ")"+ '\u00b1' + cut(offset,5) + '\u00ba' 
                    else: text2 = "    " + "K1 = (" +   str(ur) + esp2 + str(vr) + esp2 + str(wr) + ")"
                #  "    " +  u'\u03b7'+ "2 = "   + str(IntegriseAxe(array(OP),0.001))
                text3 = "   q = " + str(q) + "   Mode = " + FGmode
                ListResS.insert(END, text+text2+text3)


        def showtwin0(event):
            global twinold,twin
            #ListResS.after(10,showtwin)
            try:
                twin = [int(x) for x in ListResS.curselection()][0]
                #print "twin", twin
            except: pass
            if twin !=twinold:
                showtwin()

        def showtwinDown(event):
            global twinold,twin
            ListResS.select_clear(twin)
            twinold = twin
            twin+= 1
            try:
                ListResS.select_set(twin)
                showtwin()
            except: pass
            
        def showtwinUp(event):
            global twinold,twin
            ListResS.select_clear(twin)
            twinold = twin
            if twin>0:twin-= 1
            ListResS.select_set(twin)
            showtwin()

        def showorient(m,nbl):
            OrientMat = m
            text = "Orientation Matrix T (parent => twin) " 
            TwinBox.insert(END, text)
            text = tabify("  ",nbl) + tabify(cut(OrientMat[0,0]))  + tabify(cut(OrientMat[0,1]))  + tabify(cut(OrientMat[0,2]))
            TwinBox.insert(END, text)
            text = tabify("  ",nbl)  + tabify(cut(OrientMat[1,0])) + tabify(cut(OrientMat[1,1]))  + tabify(cut(OrientMat[1,2]))
            TwinBox.insert(END, text)
            text = tabify("  ",nbl) + tabify(cut(OrientMat[2,0])) +  tabify(cut(OrientMat[2,1])) +  tabify(cut(OrientMat[2,2]))
            TwinBox.insert(END, text)
            TwinBox.insert(END, " ")
            
            if (CRYST.syst == "Hexagonal" or CRYST.syst == "Trigonal"):
                text = "Angle between the c axes = " 
                c1 = array([0,0,1])
                c2 = dot(OrientMat,c1)
                a12 = angleVect(CRYST, vectDir(CRYST,c1),vectDir(CRYST,c2))
                a12b = 180-a12
                TwinBox.insert(END, text + cut(a12) + " Deg. (complement = " + cut(a12b) + " Deg.)" )
                TwinBox.insert(END, " ")
                
            TwinBox.insert(END, "List of equivalent rotations (ang/axis)")
            
            if CRYST.pg == CRYST.holopg():
                for roti in ListEquivRot(OrientMat, CRYST.sym(),CRYST.structTensDir):
                    ang,axis,mk = roti
                    esp = ' ' 
                    if len(axis)>0:
                        if (CRYST.syst != "Hexagonal" and CRYST.syst != "Trigonal"):
                            try: text = tabify(str(cut(ang)) + '\u00ba') + "   " +  tabify(" Axis = [" + str(axis[0]) + esp + str(axis[1]) + esp + str(axis[2])+"]")
                            except: text = ''
                        else:
                            try:
                                axis4 = ToFourIndexDir(axis)
                                text = tabify(str(cut(ang)) + '\u00ba' + "   " )+  tabify(" Axis = [" +  str(axis[0]) + esp + str(axis[1]) + esp + str(axis[2]) + "]")
                                text2 = tabify(" = 1/3 [" + str(axis4[0]) + esp + str(axis4[1]) + esp + str(axis4[2]) + esp + str(axis4[3]) + "]")
                                text = text + text2
                            except: text = ''
                        
                    else: text = "Identity"
                    if text: TwinBox.insert(END, text)
            else:
                #TwinBox.insert(END, "There are different possible orientations because the crystal is not holohedric")
                # calculate the variants by coset decomposition HoloPG/PG
                holo = CRYST.holosym()
                QuotientG = []
                for sholopg in CRYST.holosym():
                    coset = []
                    for s in CRYST.sym():
                        coset.append(dot(sholopg,s))
                    flag = 1
                    for Q in QuotientG:
                        if Inter(coset,Q):
                            flag=0
                            break
                    if flag: QuotientG.append(coset)
                #print "nb of variants holo => mero (simple cosets) = ", len(QuotientG)
                # calculate the misorientations between the twins by coset.T.coset
                listT = []
                for coseti in QuotientG:
                    si = coseti[0]
                    for cosetj in QuotientG:
                        sj = cosetj[0]
                        newT = dot(dot(inv(si),OrientMat),sj)
                        listT.append(newT)
                listTred = []
                invT = inv(OrientMat)
                for newT in listT :
                    if not IsInGroup(dot(invT,newT),CRYST.sym()): listTred.append(newT)                    
                text = "There are "+ str(len(listTred)) +" twins by double-cosets from merohedry <= holohedry / holohedry => merohedry"
                TwinBox.insert(END, text)
                TwinBox.insert(END, " ")
                indt = 0
                for newOrientMat in listTred:
                    indt+=1
                    TwinBox.insert(END, "rotation list for twin "+str(indt))
                    if det(newOrientMat)<0:
                        newOrientMat = -newOrientMat
                        textpol = "-1 x "
                    else: textpol = "  "
                    for roti in ListEquivRot(newOrientMat, CRYST.sym(),CRYST.structTensDir):
                        ang,axis,mk = roti
                        esp = ' ' 
                        if len(axis)>0:
                            if (CRYST.syst != "Hexagonal" and CRYST.syst != "Trigonal"):
                                try: text = tabify(str(cut(ang)) + " Deg.   " )+  tabify(" Axis = [" + str(axis[0]) + esp + str(axis[1]) + esp + str(axis[2]))
                                except: text = ''
                            else:
                                try:
                                    axis4 = ToFourIndexDir(axis)
                                    text = tabify(str(cut(ang)) + '\u00ba' + "   " )+  tabify(" Axis = [" +  str(axis[0]) + esp + str(axis[1]) + esp + str(axis[2]) + "]")
                                    text2 = tabify("  = 1/3 [" + str(axis4[0]) + esp + str(axis4[1]) + esp + str(axis4[2]) + esp + str(axis4[3]) + "]")
                                    text = text + text2
                                except: text = ''
                            
                        else: text = "Identity"
                        if text: TwinBox.insert(END, textpol + text)
                    TwinBox.insert(END, " ")
            
        def showtwin():
            global twinold
            #=====================================
            TwinBox.delete(0, END)
            TwinBox.insert(END, "===== Transformation matrices (Cayron, Acta Cryst. 2019), all given in the direct space ====")
            TwinBox.insert(END, " ")
            if twintype == "Dir":
                TwinBox.insert(END, "==================== Direct Mode (type I) ================================")
                r = listresultTotShearsDir[twin]
            else:
                TwinBox.insert(END, "==================== Reciprocal Mode (type II) ================================")
                r = listresultTotShearsRec[twin]
            TwinBox.insert(END, " ")            
            twinold = twin   
            shear,p,q,U,V,OA,OImin,list_s,FGmode,sHmin = r
            shear0 = shear
            transl, OP, letterABCD = list_s
            # redondance OP = OImin = point A,B,C,D le plus proche de I
            h,k,l = p
            # twin values
            #if twintype == "Rec": shear = shear*sqrt((dot(transl,dot(GMD,transl))*dot(p,dot(GME,p)))/(dot(transl,dot(GME,transl))*dot(p,dot(GMD,p))))
            OblN = arctan(shear)/ Degree
            nu1 = IntegriseAxe(array(transl),0.001)
            dirrat = IntegriseAxe(array(transl),0.001)
            ur,vr,wr = dirrat 
            if twintype == "Dir": offset = angleVect(CRYST,vectDir(CRYST,transl),vectDir(CRYST,dirrat))
            else:offset = angleVect(CRYST,vectRec(CRYST,transl),vectRec(CRYST,dirrat))
            textb = " "
            if twintype == "Dir":
                text = "Shear = " + str(cut(shear))+ "   K1 = (" + str(h) + esp2 + str(k) + esp2 + str(l) + ").   " + '\u03b7' + "1 =" + str(dirrat) 
                if (CRYST.syst == "Hexagonal" or CRYST.syst == "Trigonal"):
                    axis4 = ToFourIndexDir(dirrat)
                    textb = " = 1/3. " + " [" + str(axis4[0]) + " " + str(axis4[1]) + " " + str(axis4[2]) + " " + str(axis4[3])+ "]"
                    if offset: textb = textb + " (offset = " + cut(offset) + '\u00ba' + ")."
            else:
                if (CRYST.syst != "Hexagonal" and CRYST.syst != "Trigonal"):
                    texta = "("+ str(ur) + esp2 + str(vr) + esp2 + str(wr)+")"
                    text = "Shear = " + str(cut(shear))+ "  " + '\u03b7' + "1 = [" + str(h) + esp2 + str(k) + esp2 + str(l) + "].   " + "K1 = " + texta
                    if offset: textb = " (offset = " + cut(offset) + '\u00ba' + ")."
                    
                else:
                    axis4 = ToFourIndexDir(p) # p is a direction in this mode
                    textb = " = 1/3. " + " [" + str(axis4[0]) + " " + str(axis4[1]) + " " + str(axis4[2]) + " " + str(axis4[3])+ "]"
                    texta = "("+ str(ur) + esp2 + str(vr) + esp2 + str(wr)+")"
                    text = "Shear = " + str(cut(shear))+ "  "+'\u03b7' + "1 = [" + str(h) + esp2 + str(k) + esp2 + str(l) + "].   " + textb+ "  K1 = " + texta
                    if offset: textb = " (offset = " + cut(offset) + '\u00ba' + ")."
            TwinBox.insert(END, text+textb)
                
          
            text2 = "Crystal Obliquity (Friedel)= " + cut(OblN) + '\u00ba' + ".  Mode = " + FGmode
            text3 =  "   Twin index (q) = " + str(q)
            
            TwinBox.insert(END, text2 + text3)
            TwinBox.insert(END, " ")
    
            BpOA = transpose(array([U,V,array(OA)])) # supercell before distortion, in the parent basis, A = pt that is symtrized by H
            BpOP = transpose(array([U,V,array(OP)])) # supercell before distortion, in the parent basis, P = pt that is translated by shear
            BpOPd = transpose(array([U,V,array(OP)+array(transl)])) # supercell after distortion, in the parent basis
            BtOA = transpose(array([U,V,-array(OA)])) # supercell after distortion, in the twin basis

            # correspondence matrix
            #======================
            
            CorMat = dot(BtOA,inv(BpOP)) #see eq.9 Acta Cryst 2019, multiplied by Sigma to get integers
            if twintype == "Rec": CorMat = inv(transpose(CorMat))

            if (CRYST.struct != "FCC" and CRYST.struct != "BCC"): 
                if q >1.0001:
                    CorMatb = q*CorMat 
                    fr = "1/"+str(q)+" ."
                else:
                    CorMatb = CorMat 
                    fr = "  "
            else:
                if q >1.0001:
                    CorMatb = 2*q*CorMat 
                    fr = "1/"+str(2*q)+" ."
                else:
                    CorMatb = 2*CorMat 
                    fr = " 1/2 ."
            nbl = 0
            for ei in CorMatb:
                for ej in ei:
                    l = len(str(ej))
                    if l>nbl:nbl=l
            nbl = min(nbl,len(fr))+8
            text = "Correspondance Matrix C (twin => parent') "
            
            #if (CRYST.syst == "Hexagonal" or CRYST.syst == "Trigonal"): text = text + + " (*) see also the bottom of the page"
            TwinBox.insert(END, text)
            text = tabify("  ",nbl) + tabify(str(int(CorMatb[0,0])),nbl)  + tabify(str(int(CorMatb[0,1])),nbl)  + tabify(str(int(CorMatb[0,2])),nbl)
            TwinBox.insert(END, text)
            text = tabify(fr,nbl)  + tabify(str(int(CorMatb[1,0])),nbl)  + tabify(str(int(CorMatb[1,1])),nbl)   + tabify(str(int(CorMatb[1,2])),nbl)
            TwinBox.insert(END, text)
            text = tabify("  ",nbl) + tabify(str(int(CorMatb[2,0])),nbl) +  tabify(str(int(CorMatb[2,1])),nbl) +  tabify(str(int(CorMatb[2,2])),nbl)
            TwinBox.insert(END, text)
            TwinBox.insert(END, " ")
            
            # distortion matrix
            #======================
            DistMat = dot(BpOPd,inv(BpOP))
            if twintype == "Rec": DistMat = inv(transpose(DistMat))
            text = "Distortion Matrix F (parent => parent') "
            TwinBox.insert(END, text)
            text = tabify("  ",nbl) + tabify(cut(DistMat[0,0]))  + tabify(cut(DistMat[0,1]))  + tabify(cut(DistMat[0,2]))
            TwinBox.insert(END, text)
            text = tabify("  ",nbl)  + tabify(cut(DistMat[1,0])) + tabify(cut(DistMat[1,1]))  + tabify(cut(DistMat[1,2]))
            TwinBox.insert(END, text)
            text = tabify("  ",nbl) + tabify(cut(DistMat[2,0])) +  tabify(cut(DistMat[2,1])) +  tabify(cut(DistMat[2,2]))
            TwinBox.insert(END, text)
            TwinBox.insert(END, " ")
            sg = ShearMyFormula(DistMat,CRYST)
            print("shear my formula= ", sg)
            sgB = ShearMyFormulaB(DistMat,CRYST)
            print("shear my formula B = ", sgB)
            sg = ShearBevisCrocker(CorMat,CRYST)
            print("shear Bevis Crocker= ", sg)

            # orientation matrix
            #======================
            OrientMat = dot(BpOPd,inv(BtOA))
            if twintype == "Rec": OrientMat = inv(transpose(OrientMat))
            showorient(OrientMat,nbl)
                         

        esp = '        '
        esp2 = '  '
        try: fcalc.destroy()
        except: pass
        fcalc = Frame(frame)
        fflagDirandRec = Frame(fcalc)
        fflag = Frame(fflagDirandRec, bd=3, relief=SUNKEN)
        lt = Label(fflag, foreground="black", text="===== Twin of type I  ====").grid(row=1, column=1, columnspan = 3, sticky=W + E)
        lt = Label(fflag, foreground="black", text=" Twin plane hkl max. = ").grid(row=2, column=1, sticky=W + E)
        Labelhkl = Entry(fflag, textvariable=hkl_max,width=4)
        Labelhkl.bind("<Return>", calclistDir)
        Labelhkl.grid (row=2, column=2,  sticky='W')
        lt = Label(fflag, foreground="black", text=" Twin index q max. = ").grid(row=3, column=1, sticky=W + E)
        LabelIndex = Entry(fflag, textvariable=index_max,width=4)
        LabelIndex.bind("<Return>", calclistDir)
        LabelIndex.grid (row=3, column=2,  sticky='W')
        button = Button(fflag, foreground="blue", background="grey")
        button["text"] = "Calculate"
        button.bind("<Button>", calclistDir)
        button.grid(row=2, column=3, rowspan=2, sticky = E ,ipadx=20)
        lt = Label(fflag, foreground="black", text=" Rank the results by ").grid(row=4, column=1, sticky=W + E)
        button = Button(fflag, text="planes hkl")
        button.grid (row=4, column=2, columnspan=1)
        button.bind("<Button>", attr_rank1Dir)
        button = Button(fflag, text="shears values")
        button.grid (row=4, column=3, columnspan=1)
        button.bind("<Button>", attr_rank2Dir)                                                                                                   
        fflag.grid(row=1, column=1, sticky=N)

        fflag2 = Frame(fflagDirandRec,bd=3, relief=SUNKEN)
        lt = Label(fflag2, foreground="black", text="===== Twin of type II  ====").grid(row=1, column=1, columnspan = 3, sticky=W + E)
        lt = Label(fflag2, foreground="black", text=" Twin direction uvw max. = ").grid(row=2, column=1, sticky=W + E)
        Labelhkl = Entry(fflag2, textvariable=hkl_max,width=4)
        Labelhkl.bind("<Return>", calclistRec)
        Labelhkl.grid (row=2, column=2,  sticky='W')
        lt = Label(fflag2, foreground="black", text=" Twin index q max. = ").grid(row=3, column=1, sticky=W + E)
        LabelIndex = Entry(fflag2, textvariable=index_max,width=4)
        LabelIndex.bind("<Return>", calclistRec)
        LabelIndex.grid (row=3, column=2,  sticky='W')
        button = Button(fflag2, foreground="blue", background="grey")
        button["text"] = "Calculate"
        button.bind("<Button>", calclistRec)
        button.grid(row=2, column=3, rowspan=2, sticky = E ,ipadx=20)
        lt = Label(fflag2, foreground="black", text=" Rank the results by ").grid(row=4, column=1, sticky=W + E)
        button = Button(fflag2, text="directions uvw")
        button.grid (row=4, column=2, columnspan=1)
        button.bind("<Button>", attr_rank1Rec)
        button = Button(fflag2, text="shears values")
        button.grid (row=4, column=3, columnspan=1)
        button.bind("<Button>", attr_rank2Rec)                                                                                                   
        fflag2.grid(row=1, column=2, sticky=N)
        #build_ListResP()
        fflagDirandRec.grid(row=1, column=1, sticky=N)

        fresult = Frame(fcalc)
        fresult.grid(row=2, column=1, sticky=N)
        l = Label(fresult, text="______ List of possible twins _____")
        l.grid (row=1, column=1, columnspan=2)
        fcalc["borderwidth"] = 1
        fcalc["relief"] = SOLID
        fcalc.grid(row=1, column=2, sticky=N)
        
    #=====================================================================================================================
    # end of twinning
    #=====================================================================================================================
    
    def calc_listangles():
        global fcalc, entry_freqmax, freqmax, entry_dmin

        try: fcalc.destroy()
        except:pass

        def attr_freqmax(event):
            global fcalc, freqmax
            freqmax = string.atof(entry_freqmax.get())
            fcalc.destroy()
            build_fcalc()

        def attr_dmin(event):
            global fcalc, freqmax
            dmin = string.atof(entry_dmin.get())
            freqmax = 1. / dmin
            fcalc.destroy()
            build_fcalc()

        def attr_sym():
            global fcalc, SymFlag
            SymFlag = (SymFlag + 1) % 2
            fcalc.destroy()
            build_fcalc()

        def build_fcalc():
            global fcalc, entry_freqmax, entry_dmin
            esp = '        '
            esp2 = '  '
            Angst = '\u212B'
            try: fcalc.destroy()
            except: pass
            fcalc = Frame(frame)
            l = Label(fcalc, text="______ List of Angles _____")
            l.grid (row=0, column=1, columnspan=4)
            fflag = Frame(fcalc)
            lt = Label(fflag, foreground="black", text="dhkl min = ").grid(row=3, column=1, sticky=W + E)
            entry_dmin = Entry(fflag, width=4)
            entry_dmin.insert(0, 1. / freqmax)
            entry_dmin.bind("<Return>", attr_dmin)
            entry_dmin.grid(row=3, column=2, sticky=E + W)
            lt = Label(fflag, foreground="black", text=Angst, height=1).grid(row=3, column=3, sticky=W)
            lt = Label(fflag, foreground="black", text=" freq. max = ").grid(row=3, column=4, sticky=W + E)
            entry_freqmax = Entry(fflag, width=4)
            entry_freqmax.insert(0, freqmax)
            entry_freqmax.bind("<Return>", attr_freqmax)
            entry_freqmax.grid(row=3, column=5, sticky=E + W)
            lt = Label(fflag, foreground="black", text="1/"+Angst, height=1).grid(row=3, column=6, sticky=W)
            fflag.grid (row=1, column=1, columnspan=4)
            lt = Label(fcalc, foreground="black", text="", height=1).grid(row=2, column=1, sticky=W)
            frame2 = Frame(fcalc, width=50, borderwidth=4)
            lt = Label(frame2, foreground="black", text="Angle (" + '\u00ba' + ")"  + esp + "(hkl)1" + esp + "(hkl)2", height=1).grid(row=1, column=1, columnspan=2, sticky=W)
            Listd = Listbox(frame2, width=35, height=30)
            Listd.grid (row=2, column=2)
            s = Scrollbar(frame2)
            s.grid(row=2, column=1, sticky="NS")
            s.config(command=Listd.yview)
            Listd.config(yscrollcommand=s.set)
            for la in CRYST.listangles(freqmax):
                a, hkl1, hkl2 = la
                a = str(a)[:6]
                while len(a) <= 6:
                    a = a + '0'
                t1 = str(hkl1[0]) + esp2 + str(hkl1[1]) + esp2 + str(hkl1[2])
                t2 = str(hkl2[0]) + esp2 + str(hkl2[1]) + esp2 + str(hkl2[2])
                text = str(a + esp + t1 + esp + t2)
                Listd.insert(END, text)
                text = "---------------------------------------------------------------------------"
                Listd.insert(END, text)
            frame2.grid (row=4, column=1, columnspan=4)
            fcalc["borderwidth"] = 1
            fcalc["relief"] = SOLID
            fcalc.grid(row=1, column=2, sticky=N)

        build_fcalc()
        global fpar
        pass

    def buildFrame():
        global fpar
        global fcryst, flatt
        global list_crystsyst, list_crystpg, list_cryststruct
        global scrollsyst, scrollpg, scrollstruct
        global CRYST, AdAtom

        AdAtom = 0

        def readPos(x):
            # x = string
            # read the atomic positions, even in the a/b form
            try: r = string.atof(x)
            except:
                sp = x.find('/')
                a = string.atof(x[:sp])
                b = string.atof(x[sp + 1:])
                r = a / b
            return r

        def writePos(x):
            # x = real
            # write the atomic positions in the a/b form
            r = str(x)
            if x != 0:
                for n in range(2, 100):
                    nx = n * x
                    Intnx = int(round(nx))
                    if abs(nx - Intnx) < 0.00000001:
                        r = str(Intnx) + '/' + str(n)
                        break
            return r

        def attr_cryst(event):
            global CRYST, fcryst, flatt, fpar
            name = entry_crystname.get()
            if name != CRYST.el:
                CRYST.el = name
            r = scrollsyst.get()[0]
            syst = ListSyst[int(round(r * len(ListSyst)))]
            SeqLatt = DictLatt[syst]
            print(SeqLatt)
            if syst != CRYST.syst:
                a = string.atof(entry_crysta.get())
                if 'b' in SeqLatt: b = string.atof(entry_crystb.get())
                else: b = a
                if 'c' in SeqLatt:c = string.atof(entry_crystc.get())
                else: c = a
                if 'ag' in SeqLatt: alpha = string.atof(entry_crystalpha.get())
                else: alpha = 90
                if 'bg' in SeqLatt: beta = string.atof(entry_crystbeta.get())
                else:
                    beta = 90
                if 'cg' in SeqLatt: gamma = string.atof(entry_crystgamma.get())
                else:
                    gamma = 90
                    if syst == 'Trigonal' or syst == 'Hexagonal': gamma = 120

                CRYST.el = name
                CRYST.syst = syst
                CRYST.a, CRYST.b, CRYST.c, CRYST.alpha, CRYST.beta, CRYST.gamma = a, b, c, alpha * Degree, beta * Degree, gamma * Degree
                CRYST.listAtoms = []
            else:
                r = scrollpg.get()[0]
                listpg = DictSyst[syst]
                pg = listpg[int(round(r * len(listpg)))]
                r = scrollstruct.get()[0]
                liststruct = DictStruct[syst]
                struct = liststruct[int(round(r * len(liststruct)))]
                print(syst, pg, struct)

                a = string.atof(entry_crysta.get())
                if 'b' in SeqLatt: b = string.atof(entry_crystb.get())
                else:
                    b = a
                    entry_crystb.delete(0, END)
                    entry_crystb.insert(0, a)
                    entry_crystb["background"] = colorg
                if 'c' in SeqLatt:c = string.atof(entry_crystc.get())
                else:
                    c = a
                    entry_crystc.delete(0, END)
                    entry_crystc.insert(0, a)
                    entry_crystc["background"] = colorg
                if 'ag' in SeqLatt: alpha = string.atof(entry_crystalpha.get())
                else:
                    alpha = 90
                    entry_crystalpha.delete(0, END)
                    entry_crystalpha.insert(0, alpha)
                    entry_crystalpha["background"] = colorg
                if 'bg' in SeqLatt: beta = string.atof(entry_crystbeta.get())
                else:
                    beta = 90
                    entry_crystbeta.delete(0, END)
                    entry_crystbeta.insert(0, beta)
                    entry_crystbeta["background"] = colorg
                if 'cg' in SeqLatt: gamma = string.atof(entry_crystgamma.get())
                else:
                    gamma = 90
                    if CRYST.syst == 'Hexagonal' or CRYST.syst == 'Trigonal': gamma = 120
                    entry_crystgamma.delete(0, END)
                    entry_crystgamma.insert(0, gamma)
                    entry_crystgamma["background"] = colorg

##                C11 = string.atof(entry_C11.get())
##                C12 = string.atof(entry_C12.get())
##                C44 = string.atof(entry_C44.get())
##                C = map(lambda x:map(lambda x:0,range(6)),range(6))
##                C[0][0] = C11
##                C[1][1] = C[2][2] = C[0][0]
##                C[0][1] = C12
##                C[1][0] = C[0][2] = C[2][0] = C[1][2] = C[2][1]= C[0][1]
##                C[3][3] = C44
##                C[4][4] = C[5][5] = C[3][3]
##                stiff=array(C)
                stiff = Identity
                listAtoms0 = []
                listpos = []
                listel = []
                for entry_at in entry_atoms:
                    el = entry_at[0].get()
                    if "." in el: el = el[:el.find('.')]
                    if el not in listel:
                        listel.append(el)
                for entry_at in entry_atoms:
                    t = 0
                    el = entry_at[0].get()
                    if "." in el:
                        el = el[:el.find('.')]
                        t = 1
                    x = readPos(entry_at[1].get())
                    y = readPos(entry_at[2].get())
                    z = readPos(entry_at[3].get())
                    listpos.append(array([x, y, z]))
                    at = Atom(el, x, y, z)
                    cind = listel.index(el)
                    at.col = ListColor[cind]
                    if t: at.wick = 0
                    else: at.wick = 1
                    listAtoms0.append(at)

                listv = []
                if struct == "FCC" or struct == "F": listv = [[0.5, 0.5, 0], [0, 0.5, 0.5], [0.5, 0, 0.5]]
                elif struct == "BCC" or struct == "I": listv = [[0.5, 0.5, 0.5]]
                elif struct == "HCP": listv = [[2. / 3, 1. / 3, 0.5]]
                elif struct == "R": listv = [[2. / 3, 1. / 3, 1. / 3], [1. / 3, 2. / 3, 2. / 3],]
                elif struct == "A": listv = [[0.5, 0, 0]]
                elif struct == "B": listv = [[0, 0.5, 0]]
                elif struct == "C": listv = [[0, 0, 0.5]]

                if listv:
                    listAtoms1 = []
                    for at in listAtoms0:
                        listAtoms1.append(at)
                        for v in listv:
                            newat = transAtom(at, v)
                            newpos = array([newat.x, newat.y, newat.z])
                            nx, ny, nz = newpos
                            if (0 <= nx < 1) and (0 <= ny < 1) and (0 <= nz < 1):
                                flag = 1
                                for p in listpos: # verifie si il n a pas d atomes deja en position
                                    if add.reduce(list(map(abs, p - newpos))) < 1. / 1000000: flag = 0 # choisir un chiffre en fonction des rayons des atomes
                                if flag:
                                    listpos.append(newpos)
                                    newat = Atom(el, nx, ny, nz, newat.r, newat.col, 0) #0 pour dire que les atomes ont ete generes 
                                    listAtoms1.append(newat)

                if listv: listAtomsS = listAtoms1
                else: listAtomsS = listAtoms0

                listsym = DictPG[pg]
                # A PROGRAMMER AVEC LE GROUPE ESPACE
##                listAtoms = []
##                for at in listAtomsS: # genere tous les atomes par symetries, mais seult valable avec groupe esapce
##                    listAtoms.append(at)
##                    el = at.el
##                    pos = array([at.x, at.y, at.z])
##                    for sym in listsym:
##                        newpos = dot(sym, pos)
##                        nx, ny, nz = newpos
##                        if (0 <= nx < 1) and (0 <= ny < 1) and (0 <= nz < 1): # ne garde que ceux de la maille unite
##                            flag = 1
##                            for p in listpos: # verifie si il n a pas d atomes deja en position
##                                if add.reduce(map(abs, p - newpos)) < 1. / 1000000: flag = 0
##                            if flag:
##                                listpos.append(newpos)
##                                newat = Atom(el, nx, ny, nz, at.r, at.col, 0) #0 pour dire que les atomes ont ete generes 
##                                listAtoms.append(newat)
                                
                listAtoms = listAtomsS

                CRYST = Crystal(name, a, b, c, alpha * Degree, beta * Degree, gamma * Degree, syst, struct, pg, stiff, listAtoms)
                fpar.destroy()
                buildFrame()

            #print CRYST.toList()

        def save_cryst():
            global CRYST, entry_crystname
            attr_cryst(0)
            print("sauvegarde du crystal", CRYST.el, CRYST.a)
            os.chdir(Directory)
            CRYST.exportCryst(Directory + "\\PhasesCRYST")

        fpar = Frame(frame)

        def buildfcryst():
            global CRYST
            global fcalc, fcryst, list_crystsyst, list_crystpg, list_cryststruct, scrollsyst, scrollpg, scrollstruct
            global entry_crystname

            try: fcalc.destroy()
            except:pass

            fcryst = Frame(fpar)

            lt = Label(fcryst, foreground="black")
            lt["text"] = ""
            lt["height"] = 1
            lt.grid(row=1, column=4, sticky=W + E)

            lt = Label(fcryst)
            lt["text"] = " Crystal name "
            lt["height"] = 1
            lt.grid(row=2, column=2, sticky=E)
            entry_crystname = Entry(fcryst)
            entry_crystname.insert(0, CRYST.el)
            entry_crystname.bind("<Return>", attr_cryst)
            entry_crystname.grid(row=2, column=3, sticky=W)

            lt = Label(fcryst)
            lt["text"] = " Crystal system "
            lt["height"] = 1
            lt.grid(row=3, column=2, sticky=E)
            list_crystsyst = Listbox(fcryst, height=1)
            scrollsyst = Scrollbar(fcryst)
            scrollsyst.grid(row=3, column=4, ipadx=2, sticky="EW")
            for syst in ListSyst:
                list_crystsyst.insert(END, syst)
            scrollsyst.config(command=list_crystsyst.yview)
            list_crystsyst.config(yscrollcommand=scrollsyst.set)
            i = 0
            for syst in ListSyst:
                if CRYST.syst == syst:
                    list_crystsyst.yview_scroll(i, 'units')
                    break
                i = i + 1
            list_crystsyst.bind('<ButtonRelease-1>', attr_cryst)
            list_crystsyst.grid(row=3, column=3, sticky=W)

            lt = Label(fcryst)
            lt["text"] = " Point group "
            lt["height"] = 1
            lt.grid(row=4, column=2, sticky=E)
            list_crystpg = Listbox(fcryst, height=1)
            scrollpg = Scrollbar(fcryst)
            scrollpg.grid(row=4, column=4, ipadx=2, sticky="EW")
            for pg in DictSyst[CRYST.syst]:
                list_crystpg.insert(END, pg)
            scrollpg.config(command=list_crystpg.yview)
            list_crystpg.config(yscrollcommand=scrollpg.set)
            i = 0
            for pg in DictSyst[CRYST.syst]:
                if CRYST.pg == pg:
                    list_crystpg.yview_scroll(i, 'units')
                    break
                i = i + 1
            list_crystpg.bind('<ButtonRelease-1>', attr_cryst)
            list_crystpg.grid(row=4, column=3, sticky=W)

            lt = Label(fcryst)
            lt["text"] = " Structure "
            lt["height"] = 1
            lt.grid(row=5, column=2, sticky=E)
            list_cryststruct = Listbox(fcryst, height=1)
            scrollstruct = Scrollbar(fcryst)
            scrollstruct.grid(row=5, column=4, ipadx=2, sticky="EW")
            for struct in DictStruct[CRYST.syst]:
                list_cryststruct.insert(END, struct)
            scrollstruct.config(command=list_cryststruct.yview)
            list_cryststruct.config(yscrollcommand=scrollstruct.set)
            i = 0
            for struct in DictStruct[CRYST.syst]:
                if CRYST.struct == struct:
                    list_cryststruct.yview_scroll(i, 'units')
                    break
                i = i + 1

            list_cryststruct.grid(row=5, column=3, sticky=W)
            list_cryststruct.bind('<ButtonRelease-1>', attr_cryst)
            fcryst["borderwidth"] = 1
            fcryst["relief"] = SOLID
            fcryst.grid(row=1, column=1, sticky=E + W)

        buildfcryst()

        def buildflatt():
            global flatt
            global CRYST
            global entry_crysta, entry_crystb, entry_crystc
            global entry_crystalpha, entry_crystbeta, entry_crystgamma

            Angst = '\u212B'

            SeqLatt = DictLatt[CRYST.syst]
            flatt = Frame(fpar)
            lt = Label(flatt)
            lt["text"] = " Lattice parameters "
            lt.grid(row=1, column=1, columnspan=6, sticky=E + W)

            lt = Label(flatt)
            lt["text"] = " a "
            lt.grid(row=4, column=1, sticky=E)
            entry_crysta = Entry(flatt, width=7)
            entry_crysta.insert(0, CRYST.a)
            if 'a' in SeqLatt: entry_crysta.bind("<Return>", attr_cryst)
            else: entry_crysta["background"] = colorg
            entry_crysta.grid(row=4, column=2, sticky=E + W)
            lt = Label(flatt)
            lt["text"] = Angst
            lt.grid(row=4, column=3, sticky=W)

            lt = Label(flatt)
            lt["text"] = " b "
            lt.grid(row=5, column=1, sticky=E)
            entry_crystb = Entry(flatt, width=7)
            entry_crystb.insert(0, CRYST.b)
            if 'b' in SeqLatt: entry_crystb.bind("<Return>", attr_cryst)
            else: entry_crystb["background"] = colorg
            entry_crystb.grid(row=5, column=2, sticky=E + W)
            lt = Label(flatt)
            lt["text"] = Angst
            lt.grid(row=5, column=3, sticky=W)

            lt = Label(flatt)
            lt["text"] = " c "
            lt.grid(row=6, column=1, sticky=E)
            entry_crystc = Entry(flatt, width=7)
            entry_crystc.insert(0, CRYST.c)
            if 'c' in SeqLatt: entry_crystc.bind("<Return>", attr_cryst)
            else: entry_crystc["background"] = colorg
            entry_crystc.grid(row=6, column=2, sticky=W + E)
            lt = Label(flatt)
            lt["text"] = Angst
            lt.grid(row=6, column=3, sticky=W)

            lt = Label(flatt, font=("Symbol", 8))
            lt["text"] = " a "
            lt.grid(row=4, column=4, sticky=E)
            entry_crystalpha = Entry(flatt, width=7)
            entry_crystalpha.insert(0, round(CRYST.alpha / Degree,5))
            if 'ag' in SeqLatt: entry_crystalpha.bind("<Return>", attr_cryst)
            else: entry_crystalpha["background"] = colorg
            entry_crystalpha.grid(row=4, column=5, sticky=E + W)
            lt = Label(flatt)
            lt["text"] = '\u00ba' 
            lt.grid(row=4, column=6, sticky=W)

            lt = Label(flatt, font=("Symbol", 8))
            lt["text"] = " b "
            lt.grid(row=5, column=4, sticky=E)
            entry_crystbeta = Entry(flatt, width=7)
            entry_crystbeta.insert(0, round(CRYST.beta / Degree,5))
            if 'bg' in SeqLatt: entry_crystbeta.bind("<Return>", attr_cryst)
            else: entry_crystbeta["background"] = colorg
            entry_crystbeta.grid(row=5, column=5, sticky=E + W)
            lt = Label(flatt)
            lt["text"] = '\u00ba'
            lt.grid(row=5, column=6, sticky=W)

            lt = Label(flatt, font=("Symbol", 8))
            lt["text"] = " g "
            lt.grid(row=6, column=4, sticky=E)
            entry_crystgamma = Entry(flatt, width=7)
            entry_crystgamma.insert(0, round(CRYST.gamma / Degree,5))
            if 'cg' in SeqLatt: entry_crystgamma.bind("<Return>", attr_cryst)
            else: entry_crystgamma["background"] = colorg
            entry_crystgamma.grid(row=6, column=5, sticky=W + E)
            lt = Label(flatt)
            lt["text"] = '\u00ba'
            lt.grid(row=6, column=6, sticky=W)
            lt = Label(flatt)
            lt["text"] = ""
            lt.grid(row=7, column=1, sticky=W)

            flatt["borderwidth"] = 1
            flatt["relief"] = SOLID
            flatt.grid(row=2, column=1, sticky=E + W)

        buildflatt()

    #--

        def add_atom(event):
            global fatoms, AdAtom, CRYST
            AdAtom = 1
            fatoms.destroy()
            buildfatoms()

        def rem_atom(event):
            global fatoms, CRYST
            if not AdAtom: CRYST.listAtoms.pop()
            fatoms.destroy()
            buildfatoms()

        def buildfatoms ():
            global fatoms, entry_atoms, AdAtom, CRYST
            fatoms = Frame(fpar)
            lt = Label(fatoms)
            lt["text"] = " Atoms "
            lt.grid(row=0, column=0, sticky=E + W)

            button = Button(fatoms, foreground="black", background="grey")
            button["text"] = "+"
            button.bind("<Button>", add_atom)
            button.grid(row=0, column=2, sticky=E + W)

            button = Button(fatoms, foreground="black", background="grey")
            button["text"] = "-"
            button.bind("<Button>", rem_atom)
            button.grid(row=0, column=3, sticky=E + W)

            lt = Label(fatoms)
            lt["text"] = "  "
            lt.grid(row=1, column=0, sticky=E + W)

            print("len(CRYST.listAtoms)", len(CRYST.listAtoms))

            if not AdAtom: entry_atoms = [[None for j in range(4)] for i in range(len(CRYST.listAtoms))]
            else: entry_atoms = [[None for j in range(4)] for i in range(len(CRYST.listAtoms) + 1)]
            i = 0
            k = 0
            for at in CRYST.listAtoms:
                el, x, y, z, r, col, wick = at.el, at.x, at.y, at.z, at.r, at.col, at.wick
                j = 0
                if wick == 1:
                    f = "bold"
                else:
                    el = el + '.' #pour le codage ds wickoff
                    f = 'normal'
                entry_atoms[i][j] = Entry(fatoms, width=7, font="Times 8 " + f , foreground=col) #
                entry_atoms[i][j].insert(0, el)
                entry_atoms[i][j].bind("<Return>", attr_cryst)
                entry_atoms[i][j].grid(row=i % 20 + 2, column=j + k * len([el, x, y, z]), sticky=E + W)
                j = j + 1
                for pos in [x, y, z]:
                    entry_atoms[i][j] = Entry(fatoms, width=7, font="Times 8 " + f , foreground=col) #
                    entry_atoms[i][j].insert(0, writePos(pos))
                    entry_atoms[i][j].bind("<Return>", attr_cryst)
                    entry_atoms[i][j].grid(row=i % 20 + 2, column=j + k * len([el, x, y, z]), sticky=E + W)
                    j = j + 1
                i = i + 1
                if i % 20 == 0: k = k + 1

            if AdAtom:
                j = 0
                for info in ["X", 0, 0, 0]:
                    entry_atoms[i][j] = Entry(fatoms, width=7) #fatoms
                    entry_atoms[i][j].insert(0, info)
                    entry_atoms[i][j].bind("<Return>", attr_cryst)
                    entry_atoms[i][j].grid(row=i % 20 + 2, column=j + k * len(["X", 0, 0, 0]), sticky=E + W)
                    j = j + 1

            AdAtom = 0
            fatoms["borderwidth"] = 1
            fatoms.grid(row=4, column=1, sticky=E + W)
            fatoms["relief"] = SOLID
        buildfatoms()
    #--
##        felast = Frame(fpar)
##        lt=Label(felast)
##        lt["text"]=" STIFFNESS"
##        lt.grid(row=1,column=4,sticky = W+E)
##
##        lt=Label(felast)
##        lt["text"]=" C11 "
##        lt.grid(row=2,column=1,sticky = E)
##        entry_C11 = Entry(felast,width=10)
##        entry_C11.insert(0,CRYST.stiff[0][0])
##        entry_C11.bind("<Return>",attr_cryst)
##        entry_C11.grid(row=2,column=2,sticky = W)
##        
##        lt=Label(felast)
##        lt["text"]=" C12 "
##        lt.grid(row=2,column=3,sticky = E)
##        entry_C12 = Entry(felast,width=10)
##        entry_C12.insert(0,CRYST.stiff[0][1])
##        entry_C12.bind("<Return>",attr_cryst)
##        entry_C12.grid(row=2,column=4,sticky = W)
##        
##        lt=Label(felast)
##        lt["text"]=" C44 "
##        lt.grid(row=2,column=5,sticky = E)
##        entry_C44 = Entry(felast,width=10)
##        entry_C44.insert(0,CRYST.stiff[3][3])
##        entry_C44.bind("<Return>",attr_cryst)
##        entry_C44.grid(row=2,column=6,sticky = W)
##        
##        felast["borderwidth"]=1
##        felast.grid(row=5,column=1,sticky = E+W)
##        felast["relief"]=RAISED

        fpar["borderwidth"] = 1
        fpar["relief"] = SOLID
        fpar.grid(row=1, column=1, sticky=E + W)

        bar = Menu(root)
        filem = Menu(bar)
        filem.add_command(label="Open Crystal", command=open_cryst)
        filem.add_command(label="Save Crystal", command=save_cryst)
        bar.add_cascade(label="Files", menu=filem)
        calculatem = Menu(bar)
        calculatem.add_command(label="Distance", command=calc_dist)
        calculatem.add_command(label="Angle", command=calc_angle)
        calculatem.add_command(label="Normal direction/plane", command=calc_normal)
        calculatem.add_command(label="List of reticular distances", command=calc_listduvw)
        calculatem.add_command(label="List of interplanar distances", command=calc_listdhkl)
        calculatem.add_command(label="List of angles between planes", command=calc_listangles)
        calculatem.add_command(label="List of equivalent rotations", command=calc_listrot)
        if CRYST.syst == "Hexagonal" or CRYST.syst == "Trigonal":
            calculatem.add_command(label="Conversion 3 <>4 indices", command=calc_convert)
        calculatem.add_command(label="List of type I and II twins", command=calc_listtwins)
        calculatem.add_command(label="List of weak planes and twins", command=calc_weakhkl)
        bar.add_cascade(label="Calculate", menu=calculatem)
        try: root.config(menu=bar)
        except: pass
    buildFrame()

    frame.mainloop()

# ==============================================================================
# ==============================================================================
if __name__ == "__main__":

    m = calculeMatRot(60, [2, 1, 1])
    print("***********")
    print(m)
    print()
    print("***********")
    print(calculeRot(m))
    print(" doit etre egal a 60 deg, [211]")
    print("---------")


    element = "Cu"
    syst = 'Cubic'
    struct = "FCC"
    PG = 'm3m'
    C = [[0 for x in range(6)] for x in range(6)]
    C[0][0] = 159330.0
    C[1][1] = C[2][2] = C[0][0]
    C[0][1] = 121945.0
    C[1][0] = C[0][2] = C[2][0] = C[1][2] = C[2][1] = C[0][1]
    C[3][3] = 80943.0
    C[4][4] = C[5][5] = C[3][3]
    stiff = array(C)
    a = 3.6
    b = 3.6
    c = 3.6
    alpha = 90 * Degree
    beta = 90 * Degree
    gamma = 90 * Degree

    Cu1 = Atom("Cu", 0, 0, 0)
    Cu2 = Atom("Cu", 1. / 2, 1. / 2, 0)
    Cu3 = Atom("Cu", 1. / 2, 0, 1. / 2)
    Cu4 = Atom("Cu", 0, 1. / 2, 1. / 2)
    listAtoms = [Cu1, Cu2, Cu3, Cu4]
    Cu = Crystal(element, a, b, c, alpha, beta, gamma, syst, struct, PG, stiff, listAtoms)

    directory = getFullPath(os.path.join("Crystallography", "PhasesCRYST"))
##    Cu.exportCryst(directory)
    cryst = importCryst("Cu", directory)
##
##    print "====================",cryst.el,cryst.stiff[3][3]
##
##    Cubis=cloneCryst(cryst)
##    Cubis.a = 10
##    print Cu.a
##    print Cubis.a
##    x = Cu.metricTensDir
##    print x
##
##    v1 = vectRec(Cu,[4,3,4])
##    print v1.toOrtho()
##    v2 = vectRec(Cu,[1,0,0])
##    print v2.toOrtho()
##
##    print angleVect(Cu,v1,v2)
##    print v1.__class__.__name__
##    print v2.__class__.__name__

#---------------------------------------------------------------------------------
##    print "Si cubique en hex (3R)"
##    SiP1 = Atom("Si",0,0,0)
##    SiP2 = Atom("Si",0,0,1./4)
##    SiQ1 = Atom("Si",1./3,2./3,1./3)
##    SiQ2 = Atom("Si",1./3,2./3,1./3+1./4)
##    SiR1 = Atom("Si",2./3,1./3,2./3)
##    SiR2 = Atom("Si",2./3,1./3,2./3+1./4)
##    listAtoms = [SiP1,SiP2,SiQ1,SiQ2,SiR1,SiR2]
##    aSi = 5.43
##    a = aSi*sqrt(2)/2
##    b = a
##    c = 3*sqrt(2./3)*a
##    cryst = Crystal("Si_C3H",a,a,c,90*Degree,90*Degree,120*Degree,'Hexagonal','P','6/mmm',stiff,listAtoms)
##    #----------------------------------------------------------------

##    print "Si cubique diamond (normal)"
##    Si1 = Atom("Si",0,0,0)
##    Si2 = Atom("Si",1./4,1./4,1./4)
##    listAtoms = [Si1,Si2]
##    newlistAtoms=[]
##    for v in [[0,0,0],[0,1./2,1./2],[1./2,0,1./2],[1./2,1./2,0]]:
##        for at in listAtoms:
##            newlistAtoms.append(transAtom(at,v))
##    aSi = 5.43
##    cryst = Crystal("Si_diamond",aSi,aSi,aSi,90*Degree,90*Degree,90*Degree,'Cubic',"Diamond",'m3m',stiff,newlistAtoms)
##    cryst.exportCryst(directory)
##    #----------------------------------------------------------------
##    print "Zr_alpha"
##    pSi = 3./8
##    listAtoms =[
##        Atom("Zr",0,0,0),
##        Atom("Zr",2./3,1./3,1./2)
##        ]
##    a = 3.23
##    c = 5.15
##    cryst = Crystal("Zr_alpha",a,a,c,90*Degree,90*Degree,120*Degree,'Hexagonal','HCP','6/mmm',stiff,listAtoms)
##    cryst.exportCryst(directory)
##    #----------------------------------------------------------------
##    print "Zr_beta"
##    listAtoms = [
##        Atom("Zr",0,0,0),
##        Atom("Zr",1./2,1./2,1./2)
##        ]
##    a = 3.61
##    cryst = Crystal("Zr_beta",a,a,a,90*Degree,90*Degree,90*Degree,'Cubic','BCC','m3m',stiff,listAtoms)
##    cryst.exportCryst(directory) 
####    #----------------------------------------------------------------
##    print "Ti_alpha"
##    pSi = 3./8
##    listAtoms =[
##        Atom("Ti",0,0,0),
##        Atom("Ti",2./3,1./3,1./2)
##        ]
##    a = 2.95
##    c = 4.68
##    cryst = Crystal("Ti_alpha",a,a,c,90*Degree,90*Degree,120*Degree,'Hexagonal','HCP','6/mmm',stiff,listAtoms)
##    cryst.exportCryst(directory)
    #----------------------------------------------------------------
##    print "Ti_beta"
##    listAtoms = [
##        Atom("Ti",0,0,0),
##        Atom("Ti",1./2,1./2,1./2)
##        ]
##    a = 3.21
##    cryst = Crystal("Ti_beta",a,a,a,90*Degree,90*Degree,90*Degree,'Cubic','BCC','m3m',stiff,listAtoms)    
##    cryst.exportCryst(directory)
##    #----------------------------------------------------------------  
##    print "Fe_alpha"
##    listAtoms = [
##        Atom("Fe",0,0,0),
##        Atom("Fe",1./2,1./2,1./2)
##        ]
##    a = 2.87
##    cryst = Crystal("Fe_alpha",a,a,a,90*Degree,90*Degree,90*Degree,'Cubic','BCC','m3m',stiff,listAtoms)
##    cryst.exportCryst(directory)
##
##    #----------------------------------------------------------------
##    print "Fe_gamma"
##    listAtoms = [
##        Atom("Fe",0,0,0),
##        Atom("Fe",1./2,1./2,0),
##        Atom("Fe",0,1./2,1./2),
##        Atom("Fe",1./2,0,1./2)
##        ]
##    a = 3.58
##    cryst = Crystal("Fe_gamma",a,a,a,90*Degree,90*Degree,90*Degree,'Cubic','FCC','m3m',stiff,listAtoms)
##    cryst.exportCryst(directory)

##    print "Zirc_cubic"
##    listAtoms = [
##        Atom("Zr",0,0,0),
##        ]
##    a = 1.0 # normalement c est 5.2 mais ne marche pas avec 5.2  marche avec 1.0, avec 3.58 et pas avec 5.2 => COMPRENDRE POURQUOI!!!
##    cryst = Crystal("Zirc_cubic",a,a,a,90*Degree,90*Degree,90*Degree,'Cubic','FCC','m3m',stiff,listAtoms)
##    cryst.exportCryst(directory)

    
##        #----------------------------------------------------------------
##    print "Y2Ti2O5_Pyrochlore"
##    listAtoms = [
##        Atom("Y",0,0,0),
##        Atom("Y",1./2,1./2,0),
##        Atom("Y",1./2,0,1./2),
##        Atom("Y",0,1./2,1./2)
##        ]
##    a = 10.09
##    cryst = Crystal("Y2Ti2O5_Pyrochlore",a,a,a,90*Degree,90*Degree,90*Degree,'Cubic','FCC','m3m',stiff,listAtoms)
##    listhkl = cryst.listhkl()
##    for hkl in listhkl: print str(hkl[0])[:5],'\t',hkl[1],hkl[2],hkl[3],'\t',round(hkl[4],4)

##    print "Y2TiO5_PyrHex2"
##    listAtoms = [
##        Atom("Y",0,0,0),
##        ]
##    a = 3.66
##    c = 2*sqrt(8./3)*a
##    cryst = Crystal("Y2TiO5_PyrHex2",a,a,c,90*Degree,90*Degree,120*Degree,'Hexagonal','P','6',stiff,listAtoms)
##    #cryst.exportCryst(directory)
##    listhkl = cryst.listhkl()
##    for hkl in listhkl: print str(hkl[0])[:5],'\t',hkl[1],hkl[2],hkl[3],'\t',round(hkl[4],4)
##
##    print "Try_PyrCubic2"
##    listAtoms = [
##        Atom("Y",0,0,0),
##        Atom("Y",1./2,1./2,0),
##        Atom("Y",1./2,0,1./2),
##        Atom("Y",0,1./2,1./2)
##        ]
##    a = 2*10.09
##    cryst = Crystal("Try_PyrCubic2",a,a,a,90*Degree,90*Degree,90*Degree,'Cubic','BCC','m3m',stiff,listAtoms)
    #cryst.exportCryst(directory)
##    print "syst= ",cryst.syst,"  PG= ",cryst.pg,"  a,b,c = ",cryst.a,cryst.b,cryst.c
##    print
##    listhkl = cryst.listhkl(1,1./1.5)
##    for hkl in listhkl: print str(hkl[0])[:5],'\t',hkl[1],hkl[2],hkl[3],'\t',round(hkl[4],4)
    #-------------------------------------

##    print "Fe_hcp"
##    listAtoms = [
##        Atom("Fe",0,0,0),
##        Atom("Fe",2./3,1./3,1./2)
##        ]
##    a = 2.534
##    c = 4.138
##    cryst = Crystal("Fe_hcp",a,a,c,90*Degree,90*Degree,120*Degree,'Hexagonal','HCP','6/mmm',stiff,listAtoms) 
##    cryst.exportCryst(directory)
    
    root = Tk()
    root.title("Crystal Window")
    crystWindow(cryst,root,0)


    






