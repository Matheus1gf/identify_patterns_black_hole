import numpy as np
from spectrum import EletromagneticSpectrum
from ligth_curve import LigthCurve
from gravitational_waves import GravitationalWaves
from stellar_moviment import StellarMoviment
from particle_jets import ParticleJets

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

spectrum = EletromagneticSpectrum(X_spectrum, y_spectrum)
ligth_curve = LigthCurve(X_ligth_curve, y_ligth_curve)
gravitational_waves = GravitationalWaves(X_gravitational_waves, y_gravitational_waves)
stellar_moviment = StellarMoviment(X_stellar_moviment, y_stellar_moviment)
particle_jets = ParticleJets(X_particle_jets, y_particle_jets)

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