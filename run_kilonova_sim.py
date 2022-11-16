import os

# SMALL_CLOSE 1458

ToO_strategys = ['small_close']
maps_sizes    = ['small']
exptimes      = ['240']
ztfstrategys  = ['grg_gri_rir']#, 'grg_grg_grg', 'rgr_rir_rir']

for i in range(len(ToO_strategys)):
    ToO_strategy = ToO_strategys[i]
    maps_size = maps_sizes[i]
    exptime = exptimes[i]

    for ztfstrategy in ztfstrategys:
        command = 'python kilonova_sim.py '
        command += '-s ToO_strategy/'+ToO_strategy+'/1458.fits '
        command += '-f simsurvey_ztfstrategy/ZTF_'+ztfstrategy+'_'+exptime+'_'+maps_size+'.csv '
        command += '-o '+ToO_strategy+'_1458_'+ztfstrategy+'_'+exptime+' '
        command += '-t obsfile_noccdid '
        command += '--doSimSurvey '
        command += '--doPlots '
        command += '--doFilter '
        command += '-n 1000 '
        command += '-m Bulladynwind '
        command += '-p kilonova_models/bulladynwind/'
        print('\n----------------------------------------------')
        print('Running: ', command)
        print('----------------------------------------------\n')
        os.system(command)

