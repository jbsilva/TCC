#!/usr/bin/env python3
# -*-*- encoding: utf-8 -*-*-
# Created: Sat, 09 Jun 2018 16:54:20 -0300

"""
Gera gráficos para ilustrar a explicação sobre PCA
"""


import numpy as np
from matplotlib import pyplot as plt
from matplotlib2tikz import save as tikz_save


np.random.seed(2018)
r = 0.9

rotacao = np.pi / 6
s = np.sin(rotacao)
c = np.cos(rotacao)

X = np.random.normal(0, [0.3, 0.1], size=(50, 2)).T
R = np.array([[c, -s],
              [s, c]])
X = np.dot(R, X)

#### Figura 1 ####
fig = plt.figure(figsize=(4, 4), facecolor='w')
ax = plt.axes((0, 0, 1, 1), xticks=[], yticks=[], frameon=False)
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)

# Eixos x e y
ax.annotate(r'$x$', (-r, 0), (r, 0),
            ha='center', va='center',
            arrowprops=dict(arrowstyle='<->', color='black', lw=1))
ax.annotate(r'$y$', (0, -r), (0, r),
            ha='center', va='center',
            arrowprops=dict(arrowstyle='<->', color='black', lw=1))

# Eixos x' e y'
ax.annotate(r'$x^\prime$', (-r * c, -r * s), (r * c, r * s),
            ha='center', va='center',
            arrowprops=dict(color='black', arrowstyle='<|-|>', alpha=0.8, lw=0.5, ls='--'))
ax.annotate(r'$y^\prime$', (r * s, -r * c), (-r * s, r * c),
            ha='center', va='center',
            arrowprops=dict(color='black', arrowstyle='<|-|>', alpha=0.8, lw=0.5, ls='--'))

# Pontos do gráfico de dispersão
ax.scatter(X[0], X[1], s=25, lw=0, c='red')

# Linhas das projeções
for v in (X.T):
    vnorm = np.array([s, -c])
    d = np.dot(v, vnorm)
    v1 = v - d * vnorm
    ax.plot([v[0], v1[0]], [v[1], v1[1]], color='xkcd:baby poop', alpha=0.5, ls='dotted')

plt.savefig('pca1.pdf')
#tikz_save("pca1.tex")
plt.show()

#### Figura 2 ####
fig2 = plt.figure(figsize=(5, 5), facecolor='w')
ax = plt.axes((0, 0, 1, 1), xticks=[], yticks=[], frameon=False)
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)

# Eixos x e y
ax.annotate(r'$x$', (-r, 0), (r, 0),
            ha='center', va='center',
            arrowprops=dict(arrowstyle='<->', color='black', lw=1))
ax.annotate(r'$y$', (0, -r), (0, r),
            ha='center', va='center',
            arrowprops=dict(arrowstyle='<->', color='black', lw=1))

# Projeções
vnorm = np.array([c, s])
vnorm_row = vnorm.reshape((1, 2))
c_values = vnorm_row.dot(X)
proj = vnorm_row.T.dot(c_values)
ax.scatter(proj[0],proj[1], s=25, lw=0, c='red')

plt.savefig('pca2.pdf')
#tikz_save("pca2.tex")
plt.show()
