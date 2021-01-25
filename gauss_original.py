import numpy as np
import numpy.random as rd
import scipy as sp
from scipy import stats as st

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#尤度の計算
def calc_likelihood(data, mu, sigma, pi, K):
    likelihood = np.zeros((np.sum(n), 3))
    for k in range(K):
        likelihood[:, k] = [pi[k]*st.multivariate_normal.pdf(d, mu[k], sigma[k]) for d in data]
    return likelihood

seed = 11 #乱数固定
n = [200, 150, 150] #サンプル数
N = np.sum(n) #サンプルの和
mu_true = np.asanyarray([[0.2, 0.5], [1.2, 0.5], [2.0, 0.5]])
D = mu_true.shape[1] 
sigma_true = np.asanyarray([[[0.1,  0.085], [ 0.085, 0.1]], [[0.1, -0.085], [-0.085, 0.1]], [[0.1,  0.085], [ 0.085, 0.1]]])
c = ['r', 'g', 'b']

#正規分布に従ったサンプルを取得
rd.seed(seed) #seed読み込み
org_data = None
for i in range(3):
    print("check: ", i, mu_true[i], sigma_true[i], np.linalg.det(sigma_true[i]))
    if org_data is None:
        org_data = np.c_[st.multivariate_normal.rvs(mean=mu_true[i], cov=sigma_true[i], size=n[i]), np.ones(n[i])*i]
    else:
        org_data = np.r_[org_data, np.c_[st.multivariate_normal.rvs(mean=mu_true[i], cov=sigma_true[i], size=n[i]), np.ones(n[i])*i]]

#サンプルを2次元空間上にプロット
plt.figure(figsize=(12, 5))
for i in range(3):
    plt.scatter(org_data[org_data[:,2]==i][:,0], org_data[org_data[:,2]==i][:,1], s=30, c=c[i], alpha=0.5)
plt.savefig('/home/nishiwaki/gauss_original/real_sample.png')

# drop true cluster label
data = org_data[:,0:2].copy()

###初期化###
#混合係数の初期化
K = 3 #クラス数
pi = np.zeros(K)
for k in range(K):
    if k == K-1:
        pi[k] = 1 - np.sum(pi)
    else:
        pi[k] = 1/K
print('init pi:', pi)
#平均の初期化
max_x, min_x = np.max(data[:,0]), np.min(data[:,0])
max_y, min_y = np.max(data[:,1]), np.min(data[:,1])
mu = np.c_[rd.uniform(low=min_x, high=max_x, size=K), rd.uniform(low=min_y, high=max_y, size=K) ]
print('init mu:\n', mu)
#分散の初期化
sigma = np.asanyarray(
        [ [[0.1,  0],[ 0, 0.1]],
          [[0.1,  0],[ 0, 0.1]],
          [[0.1,  0],[ 0, 0.1]] ])
#初期値を元に対数尤度を計算
likelihood = calc_likelihood(data, mu, sigma, pi, K)
print('initial sum of log likelihood:', np.sum(np.log(likelihood)))
print('pi:\n', pi)
print('mu:\n', mu)
print('sigma:\n', sigma)

#収束条件を満たすまで更新を続ける
nframe = 1
while True:
    print('nframe:', nframe)
    plt.clf()
    
    if nframe <= 3:
        print('initial state')
        plt.scatter(data[:,0], data[:,1], s=30, c='gray', alpha=0.5, marker="+")
        for i in range(3):
            plt.scatter([mu[i, 0]], [mu[i, 1]], c=c[i], marker='o', edgecolors='k', linewidths=1)
        plt.title('initial state')
        plt.savefig('/home/nishiwaki/gauss_original/initial_state.png')

    ###Eステップ###
    #負担率計算
    likelihood = calc_likelihood(data, mu, sigma, pi, K)
    gamma = (likelihood.T/np.sum(likelihood, axis=1)).T
    N_k = [np.sum(gamma[:,k]) for k in range(K)]

    ###Mステップ###
    #混合係数の計算
    pi =  N_k/N
    #平均の計算
    tmp_mu = np.zeros((K, D))
    for k in range(K):
        for i in range(len(data)):
            tmp_mu[k] += gamma[i, k]*data[i]
        tmp_mu[k] = tmp_mu[k]/N_k[k]
    mu_prev = mu.copy()
    mu = tmp_mu.copy()
    #分散の計算
    tmp_sigma = np.zeros((K, D, D))
    for k in range(K):
        tmp_sigma[k] = np.zeros((D, D))
        for i in range(N):
            tmp = np.asanyarray(data[i]-mu[k])[:,np.newaxis]
            tmp_sigma[k] += gamma[i, k]*np.dot(tmp, tmp.T)
        tmp_sigma[k] = tmp_sigma[k]/N_k[k]
    sigma = tmp_sigma.copy()
    #対数尤度の計算
    prev_likelihood = likelihood
    likelihood = calc_likelihood(data, mu, sigma, pi, K)
    prev_sum_log_likelihood = np.sum(np.log(prev_likelihood))
    sum_log_likelihood = np.sum(np.log(likelihood))
    diff = prev_sum_log_likelihood - sum_log_likelihood
    
    #更新した値の確認
    print('sum of log likelihood:', sum_log_likelihood)
    print('diff:', diff)
    print('pi:', pi)
    print('mu:', mu)
    print('sigma:', sigma)

    for i in range(N):
        plt.scatter(data[i,0], data[i,1], s=30, c=gamma[i], alpha=0.5, marker="+")

    for i in range(K):
        ax = plt.axes()
        ax.arrow(mu_prev[i, 0], mu_prev[i, 1], mu[i, 0]-mu_prev[i, 0], mu[i, 1]-mu_prev[i, 1],
                  lw=0.8, head_width=0.02, head_length=0.02, fc='k', ec='k')
        plt.scatter([mu_prev[i, 0]], [mu_prev[i, 1]], c=c[i], marker='o', alpha=0.8)
        plt.scatter([mu[i, 0]], [mu[i, 1]], c=c[i], marker='o', edgecolors='k', linewidths=1)
    plt.title("step:{}".format(nframe))
    plt.savefig('/home/nishiwaki/gauss_original/step{}.png'.format(nframe))
    
    #収束条件
    if np.abs(diff) < 0.0001:
        break
    else:
        plt.title("iter:{}".format(nframe-3))
        nframe += 1