import subprocess

import hyperopt

min_y = 0
min_c = None
min_y_tst = 0
min_c_tst = None


def trial(hyperpm):
    global min_y, min_c, min_y_tst, min_c_tst
    
    cmd = 'python main.py --datname amazon_electronics_photo.npz --agg GAT' 
    
    cmd = 'CUDA_VISIBLE_DEVICES=5 ' + cmd
    for k in hyperpm:
        v = hyperpm[k]
        cmd += ' --' + k
        if int(v) == v:
            cmd += ' %d' % int(v)
        else:
            cmd += ' %g' % float('%.1e' % float(v))
    try:
        val, tst = eval(subprocess.check_output(cmd, shell=True))
    except subprocess.CalledProcessError:
        print('...')
        return {'loss': 0, 'status': hyperopt.STATUS_FAIL}
    print('val=%5.2f%% tst=%5.2f%% @ %s' % (val * 100, tst * 100, cmd))
    score = -val
    score_tst = -tst
    if score < min_y:
        min_y, min_c = score, cmd
        
    if score_tst < min_y_tst:
        min_y_tst, min_c_tst = score_tst, cmd
    return {'loss': score, 'status': hyperopt.STATUS_OK}


space = {'lr': hyperopt.hp.choice('lr', [0.001, 0.01, 0.025, 0.05, 0.075,  0.25, 0.5]),
         'reg': hyperopt.hp.loguniform('reg', -10, 0),
         'nlayer': hyperopt.hp.choice('nlayer', [1,2,3]),
         'ncaps': 12,
         'nhidden': hyperopt.hp.choice('nhidden', [24, 32, 40, 48, 56, 64]),
         'dropout': hyperopt.hp.uniform('dropout', 0, 1),
         'routit': 6,
         'l1':hyperopt.hp.choice('l1', [0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9]),
         'lap':hyperopt.hp.choice('lap', [0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9]),
         'att':hyperopt.hp.choice('att', [0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9]),
         'nbsz':20}
hyperopt.fmin(trial, space, algo=hyperopt.tpe.suggest, max_evals=300)
print('>>>>>>>>>> val=%5.2f%% @ %s' % (-min_y * 100, min_c))

print('>>>>>>>>>> test=%5.2f%% @ %s' % (-min_y_tst * 100, min_c_tst))
