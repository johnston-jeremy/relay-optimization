from multiprocessing import Pool, Manager
import numpy as np
import numpy.linalg as la
import cvxpy as cp
from pdb import set_trace
from sympy import expand, symbols
import tqdm
from sympy.utilities.iterables import multiset_permutations

def parse(coeffdict):
  res = []
  coeff0 = None

  for key in coeffdict:
    if key[0] == '1':
      coeff0 = coeffdict[key]
    else: 
      strlen = len(key)
      i = 0
      resinner = []
      # walk through key
      while(i < strlen):
        # check if new monomial
        if key[i] == 'x':
          i += 1
          num = ''
          exponent = ''
          # extract index
          while(i < strlen and key[i] != '*'):
            num += key[i]
            i += 1
          # check if reached the end of key
          if i == strlen:
            exponent = '1'
            pass
          # check next character
          elif key[i] == '*':
            i += 1
            # check next character. if true, then there is an exponent.
            if key[i] == '*':
              i += 1
              # extract exponent              
              while(i < strlen and key[i] != '*'):
                exponent += key[i]
                i += 1 
              i += 1
            # if there was no exponent
            else:
              exponent = '1'
        resinner.append((num, exponent, key))
      res.append(resinner)
  return res, coeff0    

def worker_allpass(inputs):
  E, H1all, H2all, params, ind = inputs
  Mr,Mb,Nu,Mu,sigma1, sigma2, Plist = params
  nsamp, nP = ind
  Pt = Pr =  Plist[nP]
  sumrate_trials = []
  H1 = H1all[nsamp]
  H2_original = H2all[nsamp]
  
  for ntrial in range(10):
    Pi = np.eye(Nu)[np.random.permutation(Nu)]
    H2 = Pi@H2_original
    Qeq, Geq = la.qr((H2@H1).T.conj(), 'complete')

    Qeq = Qeq.T.conj()
    Geq = Geq.T.conj()

    Geq_diag = np.abs(np.diag(Geq))**2
    # A = Qeq@(H1.T.conj())
    A = H1@(Qeq.T.conj())
    AHAdiag = np.array([np.linalg.norm(aa)**2 for aa in A.T])

    h = np.array([np.linalg.norm(hi2)**2*sigma1**2 for hi2 in H2])

    p = cp.Variable(Mu, pos=True)
    gs = cp.Variable(pos=True)
    # set_trace()
    C1 = [gs*AHAdiag@p + Mr*sigma1**2*gs <= Pr]
    C2 = [cp.sum(p) <= Pt]

    a  = h[:Mu]
    x = symbols('x')
    exp = 1
    for aa in a:
      exp = exp * (aa + sigma2**2*x)
    expanded = expand(exp)
    coeffdictsym = expanded.as_coefficients_dict()
    coeffdict = {}
    for i in coeffdictsym:
      coeffdict[str(i)] = coeffdictsym[i]
    res, coeff0 = parse(coeffdict)

    if coeff0 is not None:
      x = coeff0
    else:
      x = 0
    # set_trace()
    for r in res:
      term = 1
      for rr in r:
        # set_trace()
        # term *= (coeffdict[rr[2]])**(1/len(r))*cp.power(k[int(rr[0])],-int(rr[1]))
        term *= cp.power(gs,-int(rr[1]))
      x += term * coeffdict[rr[2]]
    
    # coeffs = np.array([float(coeffdict[1])] + [float(coeffdict[x**k]) for k in range(1, len(a)+1)])
    # set_trace()
    # x = coeffs[0]
    # for i in range(1, len(coeffs)):
    #   x += coeffs[i]*cp.power(gs,-i)

    obj = x*cp.prod(cp.inv_pos(p))*np.prod(1/Geq_diag[:Mu])
    constraints = C1+C2
    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve(gp=True, solver='ECOS')
    gs_opt = gs.value
    p_opt = p.value
    d_opt = Geq_diag * gs_opt
    v_opt = gs_opt*h[:Mu] + sigma2**2

    sumrate = 0.5 * np.sum([np.log2(1+di*pi/vi) for di,pi,vi in zip(d_opt,p_opt,v_opt)])
    # print(sumrate)

    W = gs_opt*np.eye(Mr)
    F = Qeq.T.conj() @ np.diag(np.sqrt(p_opt))
    for i in range(Mu):
      indexes = list(range(Mu))
      indexes.pop(i)
      interference = sum([np.abs(H2[i].T.conj()@W@H1@F[:,l])**2 for l in indexes])
      v_opt[i] += interference

    sumrate = 0.5 * np.sum([np.log2(1+di*pi/vi) for di,pi,vi in zip(d_opt,p_opt,v_opt)])
    # print(sumrate)
    # set_trace()

    sumrate_trials.append(sumrate)

  sumrate_max = np.max(sumrate_trials)

  E.append({'ind':ind, 'sumrate':sumrate_max})
  # print(len(E))

def worker_svd(inputs):
  E, H1all, H2all, params, ind = inputs
  Mr,Mb,Nu,Mu,sigma1, sigma2, Plist = params
  nsamp, nP = ind
  Pt = Pr =  Plist[nP]
  sumrate_trials = []
  H1 = H1all[nsamp]
  H2 = H2all[nsamp]

  # Q = cp.Variable(shape=(H1.shape[1],H1.shape[1]), PSD=True)
  # obj = cp.log_det(np.eye(H1.shape[0]) + (1/sigma1**2)*H1@Q@(H1.T.conj()))
  # constraints = [cp.trace(Q) <= Pt]
  # prob = cp.Problem(cp.Maximize(obj), constraints)
  # prob.solve()
  # upperbound = prob.value

  w,v = la.eig(H1.T.conj()@H1)
  q = np.zeros(w.shape[0])
  mu = [0,1000]
  while(mu[1]-mu[0]>1e-2):
    q = np.clip(np.mean(mu) - 1/w, 0, None)
    if sum(q) > Pt:
      mu[1] = np.mean(mu)
    else:
      mu[0] = np.mean(mu)
    # print(mu)
  upperbound = np.log2(la.det(np.eye(H1.shape[0]) + (1/sigma1**2)*H1@np.diag(q)@(H1.T.conj()))).real/2
  # print(upperbound/2)
  # print(sum(np.clip(np.log2(mu[0]*w), 0, None)).real/2)
  # set_trace()


  # set_trace()
  
  for ntrial in range(10):
  # for perm in multiset_permutations(np.arange(Nu)):
    Pi = np.eye(Nu)[np.random.permutation(Nu)]
    # Pi = np.eye(Nu)[perm]
    Q2, G2 = la.qr((Pi@H2).T.conj())
    Q2 = Q2.T.conj()
    G2 = G2.T.conj()

    G2_diag = np.abs(np.diag(G2))**2

    U, S, Vh = la.svd(H1)
    # A = Qeq@(H1.T.conj())
    # A = H1@(Qeq.T.conj())
    # AHAdiag = np.array([np.linalg.norm(aa)**2 for aa in A.T])

    # h = np.array([np.linalg.norm(hi2)**2*sigma1**2 for hi2 in H2])

    p = cp.Variable(Mu, pos=True)
    k = cp.Variable(Mu, pos=True)

    C1 = [k @ (cp.multiply(S,p) + sigma1**2) <= Pr]
    C2 = [cp.sum(p) <= Pt]
    # set_trace()
    nuG = 1/(S**2*G2_diag[:Mu])
    # b = nuG * sigma2**2
    x = symbols('x:'+str(Mu))
    exp = 1
    for i in range(Mu):
      expsum = 0
      for j in range(i+1):
        expsum += np.abs(G2[i,j])**2 * sigma1**2 + x[j] * sigma2**2
      exp = exp * expsum

    expanded = expand(exp)
    coeffdictsym = expanded.as_coefficients_dict()
    coeffdict = {}
    for i in coeffdictsym:
      coeffdict[str(i)] = coeffdictsym[i]
    # coeffs = np.array([float(coeffdict[1])] + [float(coeffdict[x**k]) for k in range(1, len(a)+1)])

    res, coeff0 = parse(coeffdict)

    if coeff0 is not None:
      x = coeff0
    else:
      x = 0
      
    for r in res:
      term = 1
      for rr in r:
        # set_trace()
        # term *= (coeffdict[rr[2]])**(1/len(r))*cp.power(k[int(rr[0])],-int(rr[1]))
        term *= cp.power(k[int(rr[0])],-int(rr[1]))
      x += term * coeffdict[rr[2]]
      # set_trace()
      # print(term)
      # print(coeffdict[rr[2]])
      # else:
      #   set_trace()
      #   x += coeffs[r[2]]*cp.power(k[int(r[0])],-int(r[1]))
    # print('x',x)
    obj = x*cp.prod(cp.inv_pos(p)) * np.prod(1/(S*G2_diag[:Mu]))
    # print(obj.is_dgp())
    # set_trace()
    constraints = C1+C2
    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve(gp=True, solver='ECOS')
    # set_trace()
    k_opt = k.value
    p_opt = p.value
    d_opt = G2_diag[:Mu] * k_opt * S
    v_opt = []
    for i in range(Mu):
      v_opt.append(sum([np.abs(G2[i,j])**2 * k_opt[j] * sigma1**2 for j in range(i+1)]) + sigma2**2)

    # set_trace()
    sumrate = 0.5 * np.sum([np.log2(1+di*pi/vi) for di,pi,vi in zip(d_opt,p_opt,v_opt)])
    # print(sumrate)

    # set_trace()
    W = Q2.T.conj()[:,:Mu] @ np.diag(k_opt) @ (U.T.conj()[:Mu])
    F = Vh.T.conj() @ np.diag(np.sqrt(p_opt))
    v_opt = np.zeros(Mu)
    for i in range(Mu):
      indexes = list(range(Mu))
      indexes.pop(i)
      interference = sum([np.abs(H2[i].T.conj()@W@H1@F[:,l])**2 for l in indexes])
      v_opt[i] += interference
      v_opt[i] += np.linalg.norm(H2[i].T.conj()@W)**2 * sigma1**2 + sigma2**2

    sumrate = 0.5 * np.sum([np.log2(1+di*pi/vi) for di,pi,vi in zip(d_opt,p_opt,v_opt)])
    # print(sumrate)
    # set_trace()

    sumrate_trials.append(sumrate)


  sumrate_max = np.max(sumrate_trials)

  E.append({'ind':ind, 'sumrate':sumrate_max, 'upperbound':upperbound})
  # print(sumrate_max)

def run(Mr, Mb, Nu, method, single):
  # if M is not None:
  #   Mr = Mb = M
  # else:
  #   Mr = 3 # number of relay antennas
  #   Mb = 2 # number of BS antennas

  # Nu = 5 # number of users
  Mu = np.min([Nu,Mb,Mr])
  sigma1 = sigma2 = 1

  numP = 6
  Nsamp = 100

  Plist = np.logspace(0.5,3,numP)

  # numP = 1
  # Plist = [10**2]
  params = (Mr,Mb,Nu,Mu,sigma1,sigma2,Plist)

  H1all = (np.random.randn(Nsamp, Mr, Mb) + 1j*np.random.randn(Nsamp, Mr, Mb))/np.sqrt(2)
  H2all = (np.random.randn(Nsamp, Nu, Mr) + 1j*np.random.randn(Nsamp, Nu, Mr))/np.sqrt(2)

  H1mp = []
  H2mp = []

  ind = []
  for i in range(Nsamp):
    for j in range(numP):
      ind.append((i,j))
      H1mp.append(H1all[i])
      H2mp.append(H2all[i])

  manager = Manager()
  E = manager.list()
  inputs = list(zip([E]*Nsamp*numP, [H1all]*Nsamp*numP, [H2all]*Nsamp*numP, [params]*Nsamp*numP, ind))
  
  if single == True:
    worker_svd(inputs[-1])
    worker_allpass(inputs[-1])
    return
  
  with Pool() as pool:
    if method == 'svd':
      for _ in tqdm.tqdm(pool.imap_unordered(worker_svd, inputs), total=len(inputs)):
        pass
    elif method == 'allpass':
      for _ in tqdm.tqdm(pool.imap_unordered(worker_allpass, inputs), total=len(inputs)):
        pass
  
  results = np.empty((numP, Nsamp))
  upperbound = np.empty((numP, Nsamp))
  for e in E:
    nsamp, nP = e['ind']
    results[nP, nsamp] = e['sumrate']
    if method == 'svd':
      upperbound[nP,nsamp] = e['upperbound']

  print(method, np.mean(results, axis=1))
  if method == 'svd':
    print('upper bound', np.mean(upperbound, axis=1))

  return np.mean(results, axis=1)



if __name__ == '__main__':
  # run('single')
  res = []
  
  # M = None
  Nu = 5
  # for method in ['svd','allpass']:
  for method in ['svd']:
    # for M in [2,3,4,5]:
    # for Nu in [2,5,10,15,20]:
      print('Nu =', Nu)
      res.append(run(Mr=3, Mb=2, Nu=Nu, method=method, single=False))
  print(method)
  print(res)