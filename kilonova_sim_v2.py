import numpy as np
from ligo.skymap.io import fits
from ligo.skymap.distance import parameters_to_marginal_moments

import pandas as pd
import simsurvey
import sncosmo

from astropy.coordinates import Distance
from astropy import units as u

from scipy.interpolate import RectBivariateSpline as Spline2d
from astropy.cosmology import Planck18

import sys
import os
import pickle

def load_fields_ccd(fields_file, ccd_file, mwebv=False, galactic=False, num_segs=64):
    fields_raw = np.genfromtxt(fields_file, comments='%')

    fields = {'field_id': np.array(fields_raw[:,0], dtype=int),
              'ra': fields_raw[:,1],
              'dec': fields_raw[:,2]}

    if mwebv:
        fields['mwebv'] = fields_raw[:,3]
    if galactic:
        fields['l'] = fields_raw[:,4]
        fields['b'] = fields_raw[:,5]

    ccd_corners = np.genfromtxt(ccd_file, skip_header=True)
    ccds = [ccd_corners[4*k:4*k+4, :2] for k in range(num_segs)]

    return fields, ccds


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

def bulladynwindBNS_model(dataDir=".", mdyn=0.02, mwin=0.13, phi=30, host=False):

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


def transientprop_bulladynwindBNS(mdyn=0.005, mwin=0.05, phi=30, possisDir='kilonova_models/bulladynwind', thetadist='cosine', theta_low=0, theta_upp=90, host=False):
    model = bulladynwindBNS_model(dataDir=possisDir, mdyn=mdyn, mwin=mwin, phi=phi, host=host)

    def random_parameters(redshifts, model, r_v=2., ebv_rate=0.11, **kwargs):
        amp = [] # Amplitude
        for z in redshifts:
            amp.append(10 ** (-0.4 * Planck18.distmod(z).value))

        if thetadist=='cosine':
            thetas = np.arccos(np.random.random( len(redshifts))) / np.pi * 180
        elif thetadist=='uniform':
            thetas = np.random.uniform(theta_low, theta_upp, size=len(redshifts))

        if host:
            return {
                'amplitude': np.array(amp),
                'theta': thetas,
                'hostr_v': r_v * np.ones(len(redshifts)),
                'hostebv': np.random.exponential(ebv_rate, len(redshifts))
            }
        else:
            return {
                'amplitude': np.array(amp),
                'theta': thetas
            }

    return dict(lcmodel=model, lcsimul_func=random_parameters)


def calculate_efficiency(skymap, model, strategy, pcklfilename='', ntransient=100, NPoints=2, threshold=5, doRotate=False, theta=0.0, phi=0.0,
                         survey_fields='ZTF_Fields.txt', ccd_file='ztf_rc_corners.txt', num_segs = 64,
                         inputDir='input/', zeropoint=30, dmin=0.1, dmax=1e10, sfd98_dir='input/sfd98/', rate=5e-7):
    '''
    map : skymap
    model:
    strategy: pandas dataframe
    exptime: (seconds)
    doRotate: default=False
    theta: theta rotation
    phi: phi rotation
    inputDir:
    '''

    ## HANDLE SKYMAP
    skymap, metadata = fits.read_sky_map(skymap, nest=False, distances=True)
    map_struct = {}
    map_struct["prob"] = skymap[0]
    map_struct["distmu"] = skymap[1]
    map_struct["distsigma"] = skymap[2]
    map_struct["distnorm"] = skymap[3]

    if doRotate:
        for key in map_struct.keys():
            map_struct[key] = rotate_map(map_struct[key], np.deg2rad(theta), np.deg2rad(phi))


    distmean, diststd = parameters_to_marginal_moments(map_struct["prob"],
                                                       map_struct["distmu"],
                                                       map_struct["distsigma"])

    try:
        distance_lower = Distance((distmean - 5 * diststd) * u.Mpc)
    except ValueError:
        print('Value error in Distance',(distmean - 5 * diststd),', using lower limit ', dmin)
        distance_lower = Distance(dmin * u.Mpc)
    distance_upper = Distance((distmean + 5 * diststd) * u.Mpc)
    if distance_upper.value > dmax:
        print('distance_upper.value > dmax :', distance_upper.value, '>', dmax)
        distance_upper = Distance(dmax * u.Mpc)
    z_min = distance_lower.z
    z_max = distance_upper.z

    ## SIMSURVEY
    # SURVEY PLAN . Use df for the skymap and strategy
    df = strategy
    zp = np.ones(len(df)) * zeropoint
    df['skynoise'] = 10 ** (-0.4 * (df['limmag'] - zp)) / 5

    fields, ccds = load_fields_ccd(inputDir+survey_fields, inputDir+ccd_file, num_segs=num_segs)
    df['filt'][df['filt'] == 'g'] = 'ztfg'
    df['filt'][df['filt'] == 'r'] = 'ztfr'
    df['filt'][df['filt'] == 'i'] = 'ztfi'

    plan = simsurvey.SurveyPlan(time=df['time'],
                                band=df['filt'],
                                obs_field=df['fieldid'].astype(int),
                                obs_ccd=df['ccdid'].astype(int),
                                skynoise=df['skynoise'],
                                zp=zp,
                                # comment=df['progid'],
                                fields={k: v for k, v in fields.items()
                                        if k in ['ra', 'dec', 'field_id']},
                                ccds = ccds
                                )


    # transient generator
    mjd_range = (plan.pointings['time'].min()-.5, plan.pointings['time'].min()-.1)
    dec_range = (-90,90)
    ra_range = (0, 360)

    if model=='bulladynwindBNS':
        transientprop = transientprop_bulladynwindBNS()
    else:
        print('NO MODEL DEFINED')

    tr = simsurvey.get_transient_generator([z_min, z_max],
                                          ntransient=ntransient,
                                          ratefunc=lambda z: rate,
                                          dec_range=dec_range,
                                          ra_range=ra_range,
                                          mjd_range=(mjd_range[0],
                                                     mjd_range[1]),
                                          transientprop=transientprop,
                                          sfd98_dir=sfd98_dir,
                                          skymap=map_struct,
                                          apply_mwebv=True
                                          )

    # run survey
    phase_range = (0, 4)
    survey = simsurvey.SimulSurvey(generator=tr, plan=plan,
            phase_range=phase_range, n_det=NPoints, threshold=threshold) # By default 2 detections required (n_det=2)
    
    if survey._properties['instruments'] is None:
        print('There is no hope... setting results to 0.')
        exit(0)
        
    lcs = survey.get_lightcurves(progress_bar=True, notebook=True)
    
    if lcs.meta_full == None:
        print("No lightcurves... returning.")
        exit(0)
        
    recovery_fraction = len(lcs.lcs)/ntransient
    def filterday1(lc):
        t0 = lc.meta['t0']
        mask = np.where((t0<lc['time']) & (lc['time']<t0+1))
        flux = lc['flux'][mask]
        fluxerr = lc['fluxerr'][mask]
        snr = flux/fluxerr
        if sum(snr>5)>1:
            return lc

    def filterday2(lc):
        t0 = lc.meta['t0']
        mask = np.where((t0+1<lc['time']) & (lc['time']<t0+2))
        flux = lc['flux'][mask]
        fluxerr = lc['fluxerr'][mask]
        snr = flux / fluxerr
        if sum(snr > 5) > 1:
            return lc

    def filterday3(lc):
        t0 = lc.meta['t0']
        mask = np.where((t0+2<lc['time']) & (lc['time']<t0+3))
        flux = lc['flux'][mask]
        fluxerr = lc['fluxerr'][mask]
        snr = flux/fluxerr
        if sum(snr>5)>1:
            return lc

    lcs_day1 = lcs.filter(filterday1)
    lcs_day2 = lcs.filter(filterday2)
    lcs_day3 = lcs.filter(filterday3)

    recovery_fraction_day1 = len(lcs_day1.lcs)/ntransient
    try:
        recovery_fraction_day2 = len(lcs_day2.lcs)/ntransient
    except:
        recovery_fraction_day2 = 0
    try:
        recovery_fraction_day3 = len(lcs_day3.lcs)/ntransient
    except:
        recovery_fraction_day3 = 0

    # save lcs

    pcklFile = os.path.join('lcs_out',pcklfilename)
    pickle.dump(lcs,open(pcklFile, "wb" ) )

    pickle.dump(lcs_day1,open(pcklFile[:-4]+'_day1.pkl', "wb" ) )
    pickle.dump(lcs_day2,open(pcklFile[:-4]+'_day2.pkl', "wb" ) )
    pickle.dump(lcs_day3,open(pcklFile[:-4]+'_day3.pkl', "wb" ) )

    return recovery_fraction, recovery_fraction_day1, recovery_fraction_day2, recovery_fraction_day3
        
if __name__=='__main__':
    localizations =['1045', '1045','1324', '1324','1458','1458']
    exptimes =     ['240' , '300' ,'240' , '300' ,'240' , '300']
    ztfstrategys = ['grg_gri_rir', 'grg_grg_grg', 'rgr_rir_rir']
    ntransient = 1000

    original_stdout = sys.stdout

    file_ = open('output.txt', 'w')
    file_.close()
    for i in range(len(localizations)):
        localization = localizations[i]
        exptime = exptimes[i]

        for ztfstrategy in ztfstrategys:
            #skymap = 'ToO_strategy/'+ToO_strategy+'/'+localization+'.fits'
            skymap = 'too_strategies/'+localization+'.fits'
            model = 'bulladynwindBNS'
            df_ztfstrategy = pd.read_csv('too_strategies/'+localization+'_'+exptime+'_'+ztfstrategy+'_limmag.csv')
            pcklfilename = 'lcs_'+localization+'_'+exptime+'_'+ztfstrategy+'.pkl'
            eff, eff_day1, eff_day2, eff_day3 = calculate_efficiency(skymap, model, df_ztfstrategy, pcklfilename=pcklfilename,
                                                                     ntransient=ntransient)
            print('-------------------------')
            print('| number injections:', ntransient)
            print('| skymap:', skymap)
            print('| model:', model)
            print('| ztfstrategy:', ztfstrategy)
            print('| exptime:', exptime)
            print('| Recovery fraction: ', eff)
            print('| Recovery fraction per day (1,2,3): ', eff_day1, eff_day2, eff_day3)
            print('-------------------------')

            with open('output.txt', 'a') as f:
                sys.stdout = f  # Change the standard output to the file we created.

                print('-------------------------')
                print('| number injections:', ntransient)
                print('| skymap:', skymap)
                print('| model:', model)
                print('| ztfstrategy:', ztfstrategy)
                print('| exptime:', exptime)
                print('| Recovery fraction: ', eff)
                print('| Recovery fraction per day (1,2,3): ', eff_day1, eff_day2, eff_day3)
                print('-------------------------')

                sys.stdout = original_stdout  # Reset the standard output to its original value