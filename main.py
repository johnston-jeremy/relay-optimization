from multiprocessing import Pool, Manager
import numpy as np
import numpy.linalg as la
import cvxpy as cp
from pdb import set_trace
from sympy import expand, symbols
import tqdm


def worker_allpass(inputs):
  E, H1all, H2all, params, ind = inputs
  Mr,Mb,Nu,Mu,sigma1, sigma2, Plist = params
  nsamp, nP = ind
  Pt = Pr =  Plist[nP]
  sumrate_trials = []
  H1 = H1all[nsamp]
  H2 = H2all[nsamp]

  for ntrial in range(10):
    Pi = np.eye(Nu)[np.random.permutation(Nu)]
    Qeq, Geq = la.qr((Pi@H2@H1).T.conj())
    Qeq = Qeq.T.conj()
    Geq = Geq.T.conj()

    Geq_diag = np.abs(np.diag(Geq))**2
    # A = Qeq@(H1.T.conj())
    A = H1@(Qeq.T.conj())
    AHAdiag = np.array([np.linalg.norm(aa)**2 for aa in A.T])

    h = np.array([np.linalg.norm(hi2)**2*sigma1**2 for hi2 in H2])

    p = cp.Variable(Mu, pos=True)
    gs = cp.Variable(pos=True)

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
    coeffs = np.array([float(coeffdict[1])] + [float(coeffdict[x**k]) for k in range(1, len(a)+1)])

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

    sumrate = 0.5 * np.sum([np.log2(1+di*pi/vi) for di,pi,vi in zip(d_opt,p_opt,v_opt)])

    sumrate_trials.append(sumrate)

  sumrate_max = np.max(sumrate_trials)

  E.append({'ind':ind, 'sumrate':sumrate_max})
  # print(len(E))

def run():
  Mr = 3 # number of relay antennas
  Mb = 2 # number of BS antennas
  Nu = 5 # number of users
  Mu = np.min([Nu,Mb,Mr])
  sigma1 = sigma2 = 1
  Plist = np.logspace(0.5,3,6)

  params = (Mr,Mb,Nu,Mu,sigma1,sigma2,Plist)



  # Pt = 100
  # Pr = 100
  Nsamp = 100
  H1all = (np.random.randn(Nsamp, Mr, Mb) + 1j*np.random.randn(Nsamp, Mr, Mb))/np.sqrt(2)
  H2all = (np.random.randn(Nsamp, Nu, Mr) + 1j*np.random.randn(Nsamp, Nu, Mr))/np.sqrt(2)

  H1mp = []
  H2mp = []
  numP = len(Plist)
  sumrates_vs_P = []

  ind = []
  for i in range(Nsamp):
    for j in range(numP):
      ind.append((i,j))
      H1mp.append(H1all[i])
      H2mp.append(H2all[i])

  manager = Manager()
  E = manager.list()
  inputs = zip([E]*Nsamp*numP, [H1all]*Nsamp*numP, [H2all]*Nsamp*numP, [params]*Nsamp*numP, ind)
    
  
  with Pool() as pool:
    # pool.map(worker_allpass, inputs)
    for _ in tqdm.tqdm(pool.imap_unordered(worker_allpass, inputs), total=len(inputs)):
      pass
  
  results = np.empty((numP, Nsamp))
  for e in E:
    nsamp, nP = e['ind']
    results[nP, nsamp] = e['sumrate']

  print(np.mean(results, axis=1))

if __name__ == '__main__':
  run()