import pandas as pd
import numpy as np

if __name__ == '__main__':
    time_map = {'ads':{}, 'dj':{}, 'ds':{}, 'nf':{}, 'nv':{}}
    work_map = {'ads':{}, 'dj':{}, 'ds':{}, 'nf':{}, 'nv':{}}
    
    for tk in time_map.keys():
        df = pd.read_table(tk + '_float_result', header=None, delimiter=' ')
        
        for index, row in df.iterrows():
            time_map[tk][row[0]] = row[1]
            if tk != 'nv':
                work_map[tk][row[0]] = row[2]

        df = pd.read_table(tk + '_int_result', header=None, delimiter=' ')
        for index, row in df.iterrows():
            time_map[tk][row[0]] = row[1]
            if tk != 'nv':
                work_map[tk][row[0]] = row[2]
    
    time_ad_dj, time_ad_ds, time_ad_nf, time_ad_nv = [], [], [], []
    work_ad_dj, work_ad_ds, work_ad_nf = [], [], []

    for tk in time_map['ads']:
        if tk in time_map['dj']:
            time_ad_dj.append(time_map['dj'][tk]/time_map['ads'][tk])
            work_ad_dj.append(work_map['ads'][tk]/work_map['dj'][tk])
            
        if tk in time_map['ds']:
            time_ad_ds.append(time_map['ds'][tk]/time_map['ads'][tk])
            work_ad_ds.append(work_map['ads'][tk]/work_map['ds'][tk])

        if tk in time_map['nf']:
            time_ad_nf.append(time_map['nf'][tk]/time_map['ads'][tk])
            work_ad_nf.append(work_map['ads'][tk]/work_map['nf'][tk])

        if tk in time_map['nv']:
            time_ad_nv.append(time_map['nv'][tk]/time_map['ads'][tk])

            
    print('dj time speed up ', np.mean(time_ad_dj), 'work count ', np.mean(work_ad_dj))
    print('ds time speed up ', np.mean(time_ad_ds), 'work count ', np.mean(work_ad_ds))
    print('nf time speed up ', np.mean(time_ad_nf), 'work count ', np.mean(work_ad_nf))
    print('nv time speed up ', np.mean(time_ad_nv))