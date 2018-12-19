﻿"""Hubs and authorities analysis of graph structure.
"""
#    Copyright (C) 2008-2012 by
#    Aric Hagberg <hagberg@lanl.gov>
#    Dan Schult <dschult@colgate.edu>
#    Pieter Swart <swart@lanl.gov>
#    All rights reserved.
#    BSD license.
#    NetworkX:http://networkx.github.io/
import networkx as nx
from networkx.exception import NetworkXError
__author__ = """Aric Hagberg (hagberg@lanl.gov)"""
__all__ = ['hits','bhits','hits_numpy','hits_scipy','authority_matrix','hub_matrix']

def hits(G,max_iter=100,tol=1.0e-8,nstart=None,normalized=True,weight_key="weight"):
	"""Return HITS hubs and authorities values for nodes.

	The HITS algorithm computes two numbers for a node.
	Authorities estimates the node value based on the incoming links.
	Hubs estimates the node value based on outgoing links.

	Parameters
	----------
	G : graph
	  A NetworkX graph

	max_iter : interger, optional
	  Maximum number of iterations in power method.

	tol : float, optional
	  Error tolerance used to check convergence in power method iteration.

	nstart : dictionary, optional
	  Starting value of each node for power method iteration.

	normalized : bool (default=True)
	   Normalize results by the sum of all of the values.

	weight_key:string,optional
		参照するエッジの重みのキー．

	Returns
	-------
	(hubs,authorities) : two-tuple of dictionaries
	   Two dictionaries keyed by node containing the hub and authority
	   values.

	Examples
	--------
	>>> G=nx.path_graph(4)
	>>> h,a=nx.hits(G)

	Notes
	-----
	The eigenvector calculation is done by the power iteration method
	and has no guarantee of convergence.  The iteration will stop
	after max_iter iterations or an error tolerance of
	number_of_nodes(G)*tol has been reached.

	The HITS algorithm was designed for directed graphs but this
	algorithm does not check if the input graph is directed and will
	execute on undirected graphs.

	References
	----------
	.. [1] A. Langville and C. Meyer,
	   "A survey of eigenvector methods of web information retrieval."
	   http://citeseer.ist.psu.edu/713792.html
	.. [2] Jon Kleinberg,
	   Authoritative sources in a hyperlinked environment
	   Journal of the ACM 46 (5): 604-32, 1999.
	   doi:10.1145/324133.324140.
	   http://www.cs.cornell.edu/home/kleinber/auth.pdf.
	"""
	if type(G) == nx.MultiGraph or type(G) == nx.MultiDiGraph:
		raise Exception("hits() not defined for graphs with multiedges.")
	if len(G) == 0:
		return {},{}
	# choose fixed starting vector if not given
	if nstart is None:
		h=dict.fromkeys(G,1.0/G.number_of_nodes())
	else:
		h=nstart
		# normalize starting vector
		s=1.0/sum(h.values())
		for k in h:
			h[k]*=s
	i=0
	while True: # power iteration: make up to max_iter iterations
		#hlast:最新のハブスコア
		hlast=h
		h=dict.fromkeys(hlast.keys(),0)
		a=dict.fromkeys(hlast.keys(),0)
		# this "matrix multiply" looks odd because it is
		# doing a left multiply a^T=hlast^T*G
		"""オーソリティスコア"""
		#nはリンク元、nbrはリンク先
		#オーソリティスコアはリンクされているページのハブスコアの合計（エッジ間の重みも考慮できる）
		for n in h:
			for nbr in G[n]:
				a[nbr]+=hlast[n]*G[n][nbr].get(weight_key,1)
		# now multiply h=Ga
		"""ハブスコア"""
		#nはリンク元、nbrはリンク先
		#ハブスコアはリンクしているページのオーソリティスコアの合計（エッジ間の重みも考慮できる）
		for n in h:
			for nbr in G[n]:
				h[n]+=a[nbr]*G[n][nbr].get(weight_key,1)

		# normalize vector
		s=1.0/max(h.values())#ハブスコアの最大値
		for n in h: h[n]*=s
		# normalize vector
		s=1.0/max(a.values())#オーソリティスコアの最大値
		for n in a: a[n]*=s
		# check convergence, l1 norm
		err=sum([abs(h[n]-hlast[n]) for n in h])
		if err < tol:
			break
		if i>max_iter:
			raise NetworkXError(\
			"HITS: power iteration failed to converge in %d iterations."%(i+1))
		i+=1

	if normalized:
		s = 1.0/sum(a.values())
		for n in a:
			a[n] *= s
		s = 1.0/sum(h.values())
		for n in h:
			h[n] *= s
	return h,a

def bhits(G,max_iter=100,tol=1.0e-8,nstart=None,normalized=True,weight_key="weight"):
	"""Return BHITS hubs and authorities values for nodes.
	BHITSはHITSアルゴリズムの欠点の一つであるmutualy reinforcing problem(相互リンクで不当に高いスコアの付く問題)を解決した手法
	各ノードのリンク，被リンクのホストを見るため，ホストに対する定義が必要だがFQDNにドットが2つ以下のものはそれを，3つ以上のものは3つ目までをホスト名と定義
	(実装はdomain_detect)

	Parameters
	----------
	G : graph
	  A NetworkX graph

	max_iter : interger, optional
	  Maximum number of iterations in power method.

	tol : float, optional
	  Error tolerance used to check convergence in power method iteration.

	nstart : dictionary, optional
	  Starting value of each node for power method iteration.

	normalized : bool (default=True)
	   Normalize results by the sum of all of the values.

	weight_key:string,optional
		参照するエッジの重みのキー．

	Returns
	-------
	(hubs,authorities) : two-tuple of dictionaries
	   Two dictionaries keyed by node containing the hub and authority
	   values.

	Examples
	--------
	>>> G=nx.path_graph(4)
	>>> h,a=nx.hits(G)

	Notes
	-----
	The eigenvector calculation is done by the power iteration method
	and has no guarantee of convergence.  The iteration will stop
	after max_iter iterations or an error tolerance of
	number_of_nodes(G)*tol has been reached.

	The HITS algorithm was designed for directed graphs but this
	algorithm does not check if the input graph is directed and will
	execute on undirected graphs.

	References
	----------
	.. [1] A. Langville and C. Meyer,
	   "A survey of eigenvector methods of web information retrieval."
	   http://citeseer.ist.psu.edu/713792.html
	.. [2] Jon Kleinberg,
	   Authoritative sources in a hyperlinked environment
	   Journal of the ACM 46 (5): 604-32, 1999.
	   doi:10.1145/324133.324140.a[nbr]+=hlast[n]*G[n][nbr].get(weight_key,1)/denom
	   http://www.cs.cornell.edu/home/kleinber/auth.pdf.
	"""
	if type(G) == nx.MultiGraph or type(G) == nx.MultiDiGraph:
		raise Exception("hits() not defined for graphs with multiedges.")
	if len(G) == 0:
		return {},{}
	# choose fixed starting vector if not given
	if nstart is None:
		h=dict.fromkeys(G,1.0/G.number_of_nodes())
	else:
		h=nstart
		# normalize starting vector
		s=1.0/sum(h.values())
		for k in h:
			h[k]*=s

	i=0
	while True: # power iteration: make up to max_iter iterations
		hlast=h
		h=dict.fromkeys(hlast.keys(),0)
		a=dict.fromkeys(hlast.keys(),0)
		# this "matrix multiply" looks odd because it is
		# doing a left multiply a^T=hlast^T*G
		"""オーソリティスコア"""
		#nはリンク元、nbrリンク先
		for n in h:
			for nbr in G[n]:
				"""denom=>"""
				around_n=set(G.node[n]["from_hosts"].keys())
				around_nbr=set(G.node[n]["from_hosts"].keys())
				denom=1
				for co_host in (around_n & around_nbr):
					denom += G.node[n]["from_hosts"][co_host]

				a[nbr]+=hlast[n]*G[n][nbr].get(weight_key,1)/denom
		# for n in h:
		# 	for nbr in G[n]:
		#
		# 		around_n=set(G.node[n]["from_hosts"].keys())
		# 		around_nbr=set(G.node[n]["from_hosts"].keys())
		# 		denom=1
		# 		for co_host in (around_n & around_nbr):
		# 			denom+=G.node[n]["from_hosts"][co_host]
		#
		# 		a[nbr]+=hlast[n]*G[n][nbr].get(weight_key,1)/denom
		# now multiply h=Ga
		"""ハブスコア"""
		for n in h:
			for nbr in G[n]:

				around_n=set(G.node[n]["to_hosts"].keys())
				around_nbr=set(G.node[n]["to_hosts"].keys())
				denom=1
				for co_host in (around_n & around_nbr):
					denom+=G.node[n]["to_hosts"][co_host]

				h[n]+=a[nbr]*G[n][nbr].get(weight_key,1)/denom
		# for n in h:
		# 	for nbr in G[n]:
		#
		# 		around_n=set(G.node[n]["to_hosts"].keys())
		# 		around_nbr=set(G.node[n]["to_hosts"].keys())
		# 		denom=1
		# 		for co_host in (around_n & around_nbr):
		# 			denom+=G.node[n]["to_hosts"][co_host]
		#
		# 		h[n]+=a[nbr]*G[n][nbr].get(weight_key,1)/denom
		# normalize vector
		s=1.0/max(h.values())
		for n in h: h[n]*=s
		# normalize vector
		s=1.0/max(a.values())
		for n in a: a[n]*=s
		# check convergence, l1 norm
		err=sum([abs(h[n]-hlast[n]) for n in h])
		if err < tol:
			break
		if i>max_iter:
			raise NetworkXError(\
			"HITS: power iteration failed to converge in %d iterations."%(i+1))
		i+=1
	if normalized:
		s = 1.0/sum(a.values())
		for n in a:
			a[n] *= s
		s = 1.0/sum(h.values())
		for n in h:
			h[n] *= s
	return h,a


def authority_matrix(G,nodelist=None):
	"""Return the HITS authority matrix."""
	M=nx.to_numpy_matrix(G,nodelist=nodelist)
	return M.T*M

def hub_matrix(G,nodelist=None):
	"""Return the HITS hub matrix."""
	M=nx.to_numpy_matrix(G,nodelist=nodelist)
	return M*M.T

def hits_numpy(G,normalized=True):
	"""Return HITS hubs and authorities values for nodes.

	The HITS algorithm computes two numbers for a node.
	Authorities estimates the node value based on the incoming links.
	Hubs estimates the node value based on outgoing links.

	Parameters
	----------
	G : graph
	  A NetworkX graph

	normalized : bool (default=True)
	   Normalize results by the sum of all of the values.

	Returns
	-------
	(hubs,authorities) : two-tuple of dictionaries
	   Two dictionaries keyed by node containing the hub and authority
	   values.

	Examples
	--------
	>>> G=nx.path_graph(4)
	>>> h,a=nx.hits(G)

	Notes
	-----
	The eigenvector calculation uses NumPy's interface to LAPACK.

	The HITS algorithm was designed for directed graphs but this
	algorithm does not check if the input graph is directed and will
	execute on undirected graphs.

	References
	----------
	.. [1] A. Langville and C. Meyer,
	   "A survey of eigenvector methods of web information retrieval."
	   http://citeseer.ist.psu.edu/713792.html
	.. [2] Jon Kleinberg,
	   Authoritative sources in a hyperlinked environment
	   Journal of the ACM 46 (5): 604-32, 1999.
	   doi:10.1145/324133.324140.
	   http://www.cs.cornell.edu/home/kleinber/auth.pdf.
	"""
	try:
		import numpy as np
	except ImportError:
		raise ImportError(\
			"hits_numpy() requires NumPy: http://scipy.org/")
	if len(G) == 0:
		return {},{}
	H = nx.hub_matrix(G, list(G))
	e,ev=np.linalg.eig(H)
	m=e.argsort()[-1] # index of maximum eigenvalue
	h=np.array(ev[:,m]).flatten()
	A=nx.authority_matrix(G, list(G))
	e,ev=np.linalg.eig(A)
	m=e.argsort()[-1] # index of maximum eigenvalue
	a=np.array(ev[:,m]).flatten()
	if normalized:
		h = h/h.sum()
		a = a/a.sum()
	else:
		h = h/h.max()
		a = a/a.max()
	hubs = dict(zip(G, map(float, h)))
	authorities = dict(zip(G, map(float, a)))
	return hubs,authorities

def hits_scipy(G,max_iter=100,tol=1.0e-6,normalized=True):
	"""Return HITS hubs and authorities values for nodes.

	The HITS algorithm computes two numbers for a node.
	Authorities estimates the node value based on the incoming links.
	Hubs estimates the node value based on outgoing links.

	Parameters
	----------
	G : graph
	  A NetworkX graph

	max_iter : interger, optional
	  Maximum number of iterations in power method.

	tol : float, optional
	  Error tolerance used to check convergence in power method iteration.

	nstart : dictionary, optional
	  Starting value of each node for power method iteration.

	normalized : bool (default=True)
	   Normalize results by the sum of all of the values.

	Returns
	-------
	(hubs,authorities) : two-tuple of dictionaries
	   Two dictionaries keyed by node containing the hub and authority
	   values.

	Examples
	--------
	>>> G=nx.path_graph(4)
	>>> h,a=nx.hits(G)

	Notes
	-----
	This implementation uses SciPy sparse matrices.

	The eigenvector calculation is done by the power iteration method
	and has no guarantee of convergence.  The iteration will stop
	after max_iter iterations or an error tolerance of
	number_of_nodes(G)*tol has been reached.

	The HITS algorithm was designed for directed graphs but this
	algorithm does not check if the input graph is directed and will
	execute on undirected graphs.

	References
	----------
	.. [1] A. Langville and C. Meyer,
	   "A survey of eigenvector methods of web information retrieval."
	   http://citeseer.ist.psu.edu/713792.html
	.. [2] Jon Kleinberg,
	   Authoritative sources in a hyperlinked environment
	   Journal of the ACM 46 (5): 604-632, 1999.
	   doi:10.1145/324133.324140.
	   http://www.cs.cornell.edu/home/kleinber/auth.pdf.
	"""
	try:
		import scipy.sparse
		import numpy as np
	except ImportError:
		raise ImportError(\
			"hits_scipy() requires SciPy: http://scipy.org/")
	if len(G) == 0:
		return {},{}
	M = nx.to_scipy_sparse_matrix(G, nodelist=list(G))
	(n,m)=M.shape # should be square
	A=M.T*M # authority matrix
	x=scipy.ones((n,1))/n  # initial guess
	# power iteration on authority matrix
	i=0
	while True:
		xlast=x
		x=A*x
		x=x/x.max()
		# check convergence, l1 norm
		err=scipy.absolute(x-xlast).sum()
		if err < tol:
			break
		if i>max_iter:
			raise NetworkXError(\
			"HITS: power iteration failed to converge in %d iterations."%(i+1))
		i+=1

	a=np.asarray(x).flatten()
	# h=M*a
	h=np.asarray(M*a).flatten()
	if normalized:
		h = h/h.sum()
		a = a/a.sum()
	hubs = dict(zip(G, map(float, h)))
	authorities = dict(zip(G, map(float, a)))
	return hubs,authorities

# fixture for nose tests
def setup_module(module):
	from nose import SkipTest
	try:
		import numpy
	except:
		raise SkipTest("NumPy not available")
	try:
		import scipy
	except:
		raise SkipTest("SciPy not available")
