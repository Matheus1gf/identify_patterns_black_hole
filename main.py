import numpy as np
#from spectrum import EletromagneticSpectrum
#from ligth_curve import LigthCurve
#from gravitational_waves import GravitationalWaves
#from stellar_moviment import StellarMoviment
#from particle_jets import ParticleJets
from build import Build

X_spectrum = np.random.rand(100, 10)
X_ligth_curve = np.random.rand(100, 20)
X_gravitational_waves = np.random.rand(100, 30)
X_stellar_moviment = np.random.rand(100, 15)
X_particle_jets = np.random.rand(100, 25)

y_spectrum = np.random.rand(100, 5)
y_ligth_curve = np.random.rand(100, 10)
y_gravitational_waves = np.random.rand(100, 15)
y_stellar_moviment = np.random.rand(100, 15)
y_particle_jets = np.random.rand(100, 25)

spectrum = Build(X_spectrum, y_spectrum, 'Spectrum')
ligth_curve = Build(X_ligth_curve, y_ligth_curve, 'Ligth Curve')
gravitational_waves = Build(X_gravitational_waves, y_gravitational_waves, 'Gravitational Waves')
stellar_moviment = Build(X_stellar_moviment, y_stellar_moviment, 'Stellar Moviment')
particle_jets = Build(X_particle_jets, y_particle_jets, 'Patricle Jets')

spectrum.train()
ligth_curve.train()
gravitational_waves.train()
stellar_moviment.train()
particle_jets.train()

spectrum.plot_results()
ligth_curve.plot_results()
gravitational_waves.plot_results()
stellar_moviment.plot_results()
particle_jets.plot_results()