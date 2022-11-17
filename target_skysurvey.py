import sncosmo
import numpy as np
#from .core import Transient
from skysurvey.target import Transient
from astropy.cosmology import Planck18
from scipy.interpolate import RectBivariateSpline as Spline2d

__all__ = ["bulladynwindBNS"]


class AngularTimeSeriesSource(sncosmo.Source):
    """A single-component spectral time series model.
        The spectral flux density of this model is given by
        .. math::
        F(t, \lambda) = A \\times M(t, \lambda)
        where _M_ is the flux defined on a grid in phase and wavelength
        and _A_ (amplitude) is the single free parameter of the model. The
        amplitude _A_ is a simple unitless scaling factor applied to
        whatever flux values are used to initialize the
        ``TimeSeriesSource``. Therefore, the _A_ parameter has no
        intrinsic meaning. It can only be interpreted in conjunction with
        the model values. Thus, it is meaningless to compare the _A_
        parameter between two different ``TimeSeriesSource`` instances with
        different model data.
        Parameters
        ----------
        phase : `~numpy.ndarray`
        Phases in days.
        wave : `~numpy.ndarray`
        Wavelengths in Angstroms.
        cos_theta: `~numpy.ndarray`
        Cosine of
        flux : `~numpy.ndarray`
        Model spectral flux density in erg / s / cm^2 / Angstrom.
        Must have shape ``(num_phases, num_wave, num_cos_theta)``.
        zero_before : bool, optional
        If True, flux at phases before minimum phase will be zeroed. The
        default is False, in which case the flux at such phases will be equal
        to the flux at the minimum phase (``flux[0, :]`` in the input array).
        name : str, optional
        Name of the model. Default is `None`.
        version : str, optional
        Version of the model. Default is `None`.
        """

    _param_names = ['amplitude', 'theta']
    param_names_latex = ['A', r'\theta']

    def __init__(self, phase, wave, cos_theta, flux, zero_before=True, zero_after=True, name=None,
                 version=None):
        self.name = name
        self.version = version
        self._phase = phase
        self._wave = wave
        self._cos_theta = cos_theta
        self._flux_array = flux
        self._parameters = np.array([1., 0.])
        self._current_theta = 0.
        self._zero_before = zero_before
        self._zero_after = zero_after
        self._set_theta()

    def _set_theta(self):
        logflux_ = np.zeros(self._flux_array.shape[:2])
        for k in range(len(self._phase)):
            adding = 1e-10 # Here we add 1e-10 to avoid problems with null values
            f_tmp = Spline2d(self._wave, self._cos_theta, np.log(self._flux_array[k]+adding),
                             kx=1, ky=1)
            logflux_[k] = f_tmp(self._wave, np.cos(self._parameters[1]*np.pi/180)).T

        self._model_flux = Spline2d(self._phase, self._wave, logflux_, kx=1, ky=1)


        self._current_theta = self._parameters[1]

    def _flux(self, phase, wave):
        if self._current_theta != self._parameters[1]:
            self._set_theta()
        f = self._parameters[0] * (np.exp(self._model_flux(phase, wave)))
        if self._zero_before:
            mask = np.atleast_1d(phase) < self.minphase()
            f[mask, :] = 0.
        if self._zero_after:
            mask = np.atleast_1d(phase) > self.maxphase()
            f[mask, :] = 0.
        return f

def bulladynwindBNS_model(dataDir=".", mdyn=0.005, mwin=0.05, phi=30, host=False):

    l = dataDir + '/nsns_nph1.0e+06_mejdyn' + '{:.3f}'.format(mdyn) + '_mejwind' + '{:.3f}'.format(
            mwin) + '_phi' + '{:.0f}'.format(phi) + '.txt'
    f = open(l)
    lines = f.readlines()

    nobs = int(lines[0])
    nwave = float(lines[1])
    line3 = (lines[2]).split(' ')
    ntime = int(line3[0])
    t_i = float(line3[1])
    t_f = float(line3[2])

    cos_theta = np.linspace(0, 1, nobs)  # 11 viewing angles
    phase = np.linspace(t_i, t_f, ntime)  # epochs

    file_ = np.genfromtxt(l, skip_header=3)

    wave = file_[0:int(nwave), 0]
    flux = []
    for i in range(int(nobs)):
        flux.append(file_[i * int(nwave):i * int(nwave) + int(nwave), 1:])
    flux = np.array(flux).T

    phase = np.linspace(t_i, t_f, len(flux.T[0][0]))  # epochs

    source = AngularTimeSeriesSource(phase, wave, cos_theta, flux)
    dust = sncosmo.CCM89Dust()
    if host:
        return sncosmo.Model(source=source,effects=[dust], effect_names=['MW', 'host'], effect_frames=['obs', 'rest'])
    else:
        return sncosmo.Model(source=source,effects=[dust], effect_names=['MW'], effect_frames=['obs'])


class bulladynwindBNS( Transient ):

    _KIND = "bulladynwindBNS"
    possisDir = 'kilonova_models/bulladynwind'
    _TEMPLATE = bulladynwindBNS_model(dataDir=possisDir)

    _RATE = 1e-3


    # {'model': func, 'prop': dict, 'input':, 'as':}
    _MODEL = dict(redshift={"param": {"zmax": 0.051}, "as": "z"},
                  t0={"model": np.random.uniform,
                      "param": {"low": 59883.5, "high": 59884}},  # the 't0' parameters
                  theta={"model": "theta_uniform_cosine", "input": ['z']},
                  amplitude={"model": "amp_dist", "input": ['z']},
                  # This you need to match with the survey
                  radec={"model": "random",
                         "as": ["ra", "dec"]}
                  )

    def theta_uniform_cosine(cls, z):
        return np.arccos(np.random.random(len(z))) / np.pi * 180

    def amp_dist(cls, z):
        amp = [] # Amplitude
        for z_ in z:
            amp.append(10 ** (-0.4 * Planck18.distmod(z_).value))

        return amp
