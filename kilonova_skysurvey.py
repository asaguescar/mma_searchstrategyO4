import pandas as pd
from skysurvey.survey import ZTF
import sys
import numpy as np
from skysurvey import survey, target
from skysurvey import DataSet

from target_skysurvey import bulladynwindBNS

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
    df = strategy
    df['zp'] = np.ones(len(df)) * zeropoint
    df['gain'] = np.ones(len(df)) * 1
    df['skynoise'] = 10 ** (-0.4 * (df['limmag'] - df['zp'])) / 5
    df['filt'][df['filt'] == 'g'] = 'ztfg'
    df['filt'][df['filt'] == 'r'] = 'ztfr'
    df['filt'][df['filt'] == 'i'] = 'ztfi'
    df['mjd'] = df['time']
    df['band'] = df['filt']
    df = df.groupby(['mjd', 'band'], as_index=False).mean()
    #print(df)
    ztf = ZTF(data=df)

    # Injections
    kne = bulladynwindBNS.from_draw(ntransient)
    #print(kne.data)
    #print(ztf.data)

    dset = DataSet.from_targets_and_survey(kne, ztf)
    ndetection = dset.get_ndetection(per_band=False)
    detindex = ndetection[ndetection>2].index
    recovery_fraction = len(detindex)/ntransient

    print('\nrecovery_fraction:', recovery_fraction)

    return recovery_fraction, recovery_fraction_day1, recovery_fraction_day2, recovery_fraction_day3


if __name__=='__main__':
    localizations =['1045', '1045','1324', '1324','1458','1458']
    exptimes =     ['240' , '300' ,'240' , '300' ,'240' , '300']
    ztfstrategys = ['grg_gri_rir', 'grg_grg_grg', 'rgr_rir_rir']
    ntransient = 10000

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