import numpy as np

mgc_dim = 40
bap_dim =26

def MCD(mgc1, mgc2):
	tmp = (mgc1 - mgc2)**2
	tmp = 10*np.sqrt(2)/np.log(10)*np.sqrt(np.sum(tmp, axis=1))
	MCD = np.mean(tmp)
	return MCD

def RMSE(lf01, lf02):
	tmp = np.mean((np.exp(lf01) - np.exp(lf02))**2)
	return np.sqrt(tmp)

def bap_err(bap1, bap2):
	tmp = (bap1-bap2)**2
	tmp = 10*np.sqrt(2)/np.log(10)*np.sqrt(np.sum(tmp,axis=1))
	err = np.mean(tmp)
	return err
	
def vu_err(vu1, vu2):
	vu2 [vu2 < 0.5] = 0
	vu2 [vu2 >= 0.5] =1
	err = np.mean(np.abs(vu1 - vu2))

	return err

def get_err(ori, pre):
	mgc1 = ori[:,1:mgc_dim]
	mgc2 = pre[:,1:mgc_dim]

	f01 = ori[:, mgc_dim*3]
	f02 = pre[:, mgc_dim*3]

	bap1 = ori[:, (mgc_dim*3+3+1):(mgc_dim*3+3+25)]
	bap2 = pre[:, (mgc_dim*3+3+1):(mgc_dim*3+3+25)]

	vu1 = ori[:, -1]
	vu2 = pre[:, -1]

	return MCD(mgc1, mgc2), RMSE(f01, f02), bap_err(bap1, bap2), vu_err(vu1, vu2)
