"""Physics-agnostic near-tangency regression (no literature output planted)."""
import numpy as np
import pytest
from mpmath import mp

from cualni_cryst.ptmc_adapter import _stable_quadratic_roots, solve_affine_ptmc_parameters
from cualni_cryst.ptmc_root_isolation import (
    resolve_bounded_habit_roots, _determinant_polynomial,
)

# Independent 85-digit coefficients RECORDED by the independent audit on
# encoded 2H-derived synthetic perturbations. Only used to check topology,
# not to construct production solutions or change physical input values.
AUDIT_ROWS = (
  ('-1e-6',2,'0.0003106615983651161441098376210137018845026160971107597363854854084754437854665222217571','-0.001242648882347845173276403446427935875572317477255984199787188457061648555536865458398','0.001242648882347845437649238563954430798731057482159757423821082378399137794897784707827'),
  ('-1e-8',2,'0.0003106637287409908050162202983158283149498542682771606302230273523369092109837131604915','-0.001242654939817420820362672684325830046123968129818306503681636142058866979701596345111','0.001242654939817420872054745712802254806265846494051283169153436938787714200194960070518'),
  ('-1e-10',2,'0.0003106637500358312774348156929759562836674095064816324687194043323336455172450985452859','-0.001242655000391856038987268793015414312710482724201709268940847621331702037754871721321','0.001242655000391856108088283262655964858200936219347445568784928840460332739056186697316'),
  ('0',0,'0.000310663750250929712894576508938527962571388523478801538924923777404566743230870547138','-0.001242655001003718783200507956787724444837811184131036471592258651678356316057606028631','0.001242655001003718904075628563374011185577695768275132904479166493806685072825131419667'),
  ('1e-10',0,'0.0003106637504660278371385705093155978458889740900676060089078415005707452245407307147204','-0.001242655001615580905818143044894321367993601746767296440866296660825875329828662080285','0.001242655001615581078322257647513552606137566666434827347650810097669966199545706043682'),
  ('1e-8',0,'0.0003106637717606884135591375054968961660661013730491466635110161801463075580357394958113','-0.001242655062190011218596291473941611849348777247687515363479018356803547407174511762523','0.001242655062190011176871253709021062828535827803682965604364711930916484329841422632817'),
  ('1e-6',0,'0.0003106659003348535210386302155342806651707691951494397648291668117587026754317139031472','-0.00124266111960681911039538664804025902061566095588315020411551837223799454800681174648','0.001242661119606819159682041747506334886383258641440684330076180157029581926182079488054'),
)

@pytest.mark.parametrize('label,expected,c0,c1,c2', AUDIT_ROWS)
def test_independent_quadratic_discriminants_have_expected_topology(label,expected,c0,c1,c2):
    with mp.workdps(100):
        a,b,c=mp.mpf(c0),mp.mpf(c1),mp.mpf(c2)
        disc=b*b-4*a*c
        assert (disc > 0)==(expected==2),label
        assert (disc < 0)==(expected==0),label
        # Comparison is with exact text coefficients; their decimal values
        # were generated from independently encoded binary64 matrices.

@pytest.mark.parametrize('coeffs',[
   (0.25,-1.0,1.0),
   (0.25000001,-1.0,1.0),
   (0.24999999,-1.0,1.0),
   (0.00031066377176068826,-0.0012426550621900104,0.0012426550621900106),
])
def test_fast_solver_never_clamps_negative_discriminant(coeffs):
    c0,c1,c2=coeffs
    disc=c1*c1-4*c2*c0
    roots=_stable_quadratic_roots(*coeffs,tolerance=1e-10)
    if disc<0: assert roots==()
    elif disc==0: assert len(roots)==1
    else: assert len(roots)==2


def test_analytic_double_root_has_one_habit_fraction():
    F=np.diag([1.,1.,1.3]);F[0,1]=-0.5
    D=np.zeros((3,3));D[0,1]=1.0
    assert resolve_bounded_habit_roots(F,D)==(0.5,)
    result=solve_affine_ptmc_parameters(F,D,domain=(0.,1.))
    assert result.discrete_roots==(0.5,) and not result.continuum


def test_encoded_pencil_near_tangent_root_topology():
    # Independent synthetic rank-one state. Not a real Cu-Al-Ni specimen.
    F=np.diag([1.03791532,1.0309579,1.14501251])
    D=np.outer(np.array([0.83942037,0.7860128,-0.33211771]),
               np.array([0.43941288,0.76523371,1.10459369]))*0.30
    # Scaling is found independently from an interior local maximum; no
    # production-predicted lambda is used to define these inputs.
    from scipy.optimize import minimize_scalar
    def middle(f):return float(np.sort(np.linalg.svd(F+f*D,compute_uv=False))[1])
    opt=minimize_scalar(lambda f:-middle(f),bounds=(0.15,0.85),method='bounded',options={'xatol':1e-13})
    if not opt.success or not (.15<opt.x<.85):
        pytest.skip('This synthetic orientation does not yield a bounded interior peak')
    scale=middle(opt.x)
    for eps in (-1e-6,-1e-8,-1e-10,1e-10,1e-8,1e-6):
        A=F/(scale*(1+eps));B=D/(scale*(1+eps))
        high=resolve_bounded_habit_roots(A,B)
        prod=solve_affine_ptmc_parameters(A,B,domain=(0.,1.))
        assert len(prod.discrete_roots)==len(high), (eps,prod.discrete_roots,high)
        assert np.allclose(prod.discrete_roots,high,atol=2e-7,rtol=0)
        if eps<0: assert len(high)==2
        else: assert high==()


def test_root_isolator_never_accepts_wrong_singular_value():
    F=np.diag([.85,1.1,1.3]);D=np.diag([.3,0,0])
    # One determinant zero at f=0.5 is for the lowest singular value,
    # while the middle singular value remains 1.1: NOT a habit root.
    assert resolve_bounded_habit_roots(F,D)==()


@pytest.mark.parametrize('scale',[2.0**-8, 1.0, 2.0**8])
def test_exact_tangent_invariant_under_binary_power_scaling(scale):
    F=np.diag([1.,1.,1.25]);F[0,1]=-0.5
    D=np.zeros((3,3));D[0,1]=1.0
    # Power-of-two scale transformations are exact in binary64. This tests
    # the determinant and target singular value under joint rescaling.
    roots=resolve_bounded_habit_roots(scale*F,scale*D,
                   dilatational_factor=scale)
    assert roots==(0.5,)


def test_bounded_endpoint_root_is_not_discarded():
    F=np.diag([0.8,0.9,1.2]);D=np.zeros((3,3));D[0,0]=0.2
    roots=resolve_bounded_habit_roots(F,D)
    # The root is close to one and may shift at the binary64 encoding level.
    assert len(roots)==1 and abs(roots[0]-1.0)<1e-12


def test_exact_continuum_is_not_misreported_as_a_discrete_tangent():
    from cualni_cryst.ptmc_root_isolation import UnresolvedPTMCRoot
    F=np.diag([0.9,1.0,1.1]);D=np.zeros((3,3));D[0,0]=0.05
    with pytest.raises(UnresolvedPTMCRoot,match='continuum'):
        resolve_bounded_habit_roots(F,D)
