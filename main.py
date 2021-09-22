import numpy as np
import numpy.linalg as la
import cvxpy as cp
from pdb import set_trace
from sympy import expand, symbols, Integer, Mul, Sum, Float

Mr = 3 # number of relay antennas
Mb = 2 # number of BS antennas
Nu = 5 # number of users
Mu = np.min([Nu,Mb,Mr])

Pt = 100
Pr = 100

H1 = np.random.randn(Mr, Mb)
H2 = np.random.randn(Nu, Mr)
sumratesmax = []
for niter1 in range(100):
  print(niter1)
  sumrates = []
  for niter2 in range(10):
    
    Pi = np.eye(Nu)[np.random.permutation(Nu)]

    sigma1 = 1
    sigma2 = 1

    Qeq, Geq = la.qr((Pi@H2@H1).T.conj())
    Qeq = Qeq.T.conj()
    Geq = Geq.T.conj()
    # set_trace()

    Geq_diag = np.abs(np.diag(Geq))**2
    # A = Qeq@(H1.T.conj())
    A = H1@(Qeq.T.conj())

    p = cp.Variable(Mu, pos=True)
    gs = cp.Variable(pos=True)

    d = gs * Geq_diag
    h = np.array([np.linalg.norm(hi2)**2*sigma1**2 for hi2 in H2])

    AHAdiag = np.diag(A.T.conj() @ A)

    C1 = [gs*AHAdiag@p + Mr*sigma1**2*gs <= Pr]
    C2 = [cp.sum(p) <= Pt]

    a  = Geq_diag[:Mu] * h[:Mu]
    b = Geq_diag * sigma2**2
    x = symbols('x')
    exp = 1
    for aa,bb in zip(a,b):
      exp = exp * (aa + bb*x)
    expanded = expand(exp)
    coeffdict = expanded.as_coefficients_dict()
    coeffs = np.array([np.float(coeffdict[1])] + [np.float(coeffdict[x**k]) for k in range(1, len(a)+1)])

    x = coeffs[0]
    for i in range(1, len(a)+1):
      x += coeffs[i]*cp.power(gs,-i)

    obj = x*cp.prod(cp.inv_pos(p))
    constraints = C1+C2
    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve(gp=True, solver='ECOS')
    gs_opt = gs.value
    p_opt = p.value
    d_opt = Geq_diag * gs_opt
    v_opt = gs_opt*h[:Mu] + sigma2**2

    sumrates.append(0.5 * np.sum([np.log2(1+di*pi/vi) for di,pi,vi in zip(d_opt,p_opt,v_opt)]))
    # print(len(d_opt),len(p_opt),len(v_opt))
  sumratesmax.append(np.max(sumrates))
  
print(sumratesmax)
print(np.mean(sumratesmax))