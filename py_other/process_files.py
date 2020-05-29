#! /usr/bin/python2

import multiprocessing
import numpy as np 
import os
try:
    import cPickle as pickle
except:
    import pickle

"""
How to use
files_all = process_files.crawl4pickles(['/mnt/data/mixture_gp_tests/'])
paramsets = process_files.get_data(files_all,desired_params)
process_files.process_data(paramsets)

All data is stored in paramsets
"""



def crawlfolder(folders_path,termination='.p',verbose=True):
    """
    input 
        folders_path (list) with the path to all folders to crawl
    output
        files_all (list) all files with termination <termination>
    """

    folders = list()
    folders = folders+folders_path
    files_all = list()

    nfolders = 0
    nt = len(termination)
    while len(folders) != 0:
        cfolder = folders.pop(0) #get first item from folders and remove it
        nfolders += 1
        #check for subfolders
        _folders = [cfolder+'/'+sfolder for sfolder in os.listdir(cfolder) if os.path.isdir(cfolder+'/'+sfolder)]
        folders = folders+_folders
        #check for <termination> files
        _files = [cfolder+'/'+f for f in os.listdir(cfolder) if (f[-nt:]==termination)] #retrieve all *<termination> files
        files_all = files_all +_files
        
    if verbose: print('folders searched: ',nfolders); print('files found: ',len(files_all))
    return files_all


def get_data(files_all,desired_params):
    # create dictionary with all desired combinations of parameters
    keys_list = list()
    values_list = list()
    nperm_list = list()

    for key in sorted(desired_params):
        if desired_params[key] is not None:
            keys_list.append(key)
            values_list.append(desired_params[key])
            nperm_list.append(len(desired_params[key]))

    totalcomb = np.prod(nperm_list)
    nparam = len(nperm_list)
    comb = [[0 for k in range(nparam)] for l in range(totalcomb)]

    nrep = 1; nr   = 0
    for i in range(nparam):
        n_elem_i = nperm_list[i]
        k = 0
        for j in range(totalcomb):
            k = k%n_elem_i
            comb[j][i] = values_list[i][k]
            nr  += 1
            if nr == nrep:
                k += 1
                nr = 0
        nrep = nrep*n_elem_i
        
    paramsets = {'data':list(), 'comb':comb, 'keys':keys_list}
    
    # filter files not in desired params
    ff = FilterFiles(desired_params)
    files_all = ff.run(files_all) 

    # read and sort files
    readf = ReadFiles(paramsets)
    paramsets['data'] = readf.run(files_all)

    return paramsets

def process_data(paramsets):
    compute = Compute(paramsets)
    compute.run()

class ReadFiles:
    def __init__(self,paramsets):
        self.paramsets = paramsets

    def __call__(self,file_path): #check keys
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)

            # get index in paramset
            data_param = list()
            for key in self.paramsets['keys']:
                data_param.append(data['mcl'][key])
            cindex = self.paramsets['comb'].index(data_param)

            #process data
            ground_truth_poses = data['lrf']
            wifi_poses = data['wifi']

            gt_pos  = np.asarray([[p.pose.pose.position.x,p.pose.pose.position.y] for p in ground_truth_poses])
            wifi_pos  = np.asarray([[p.pose.position.x,p.pose.position.y] for p in wifi_poses])

            gt_time = [p.header.stamp.secs+p.header.stamp.nsecs/1e9 for p in ground_truth_poses]
            wifi_time = [p.header.stamp.secs+p.header.stamp.nsecs/1e9 for p in wifi_poses]

            _temp = [next((i for i,time1 in enumerate(gt_time) if time1>wtime), None) for wtime in wifi_time]
            index_next = [tmp for tmp in _temp if tmp is not None]
            index_prev = [ip-1 for ip in index_next]

            dt  = np.asarray([gt_time[inext]-gt_time[iprev] for inext, iprev in zip(index_next,index_prev)])
            dta = np.asarray([gt_time[inext]-wtime for inext, wtime in zip(index_next,wifi_time)])/dt
            dtb = np.asarray([wtime-gt_time[iprev] for iprev, wtime in zip(index_prev,wifi_time)])/dt
            _gt_pos = np.transpose(dta*np.transpose(gt_pos[index_prev])+dtb*np.transpose(gt_pos[index_next]))
            _wf_pos = wifi_pos[:_gt_pos.shape[0],:]
            _time   = wifi_time[:_gt_pos.shape[0]]

            errors = np.sum((_gt_pos-_wf_pos)**2,axis=1)**.5
        except:
            print('[WARN] Failed processing: '+file_path)
        
        return [cindex, file_path, errors, _gt_pos, _wf_pos, _time]
        
    def run(self, files_all, pool_workers=100):
        pool = multiprocessing.Pool(pool_workers)
        
        comb_data = list()
        for i in range(len(self.paramsets['comb'])):
            comb_data.append({'file_name':list(),'error':list(), 'wifi':list(), 'gt':list(), 'time':list(), 'nfiles':0})
        
        for results in pool.map(self, files_all):
            if results is not None:
                cindex = results[0]
                comb_data[cindex]['file_name'].append(results[1])
                comb_data[cindex]['error'].append(results[2])
                comb_data[cindex]['gt'].append(results[3])
                comb_data[cindex]['wifi'].append(results[4])
                comb_data[cindex]['time'].append(results[5])
                comb_data[cindex]['nfiles'] += 1

        pool.close()
        pool.join()
        
        return comb_data

class FilterFiles:
    def __init__(self,desired_params,**kwargs):
        self.desired_params = desired_params
        
    def __call__(self,file_path): #check keys
        add_file = 1
        with open(file_path,'rb') as f:
            data = pickle.load(f)

        for key, value in self.desired_params.items():
            if value is not None:
                if key in data['mcl']:
                    if not data['mcl'][key] in value: add_file = 0
                else:
                    add_file = 0
        if add_file: return file_path
        else: return ''

    def run(self, files_all, pool_workers=100):
        pool = multiprocessing.Pool(pool_workers)
        total = list()
        for res in pool.map(self, files_all):
            total.append(res)
            
        pool.close()
        pool.join()

        return list(filter(None,total))

class Compute:
    def __init__(self,paramsets,**kwargs):
        self.paramsets = paramsets
        self.tmin = kwargs.get('tmin',50)
        self.tmax = kwargs.get('tmax',-10)
        self.thr_error = kwargs.get('thr_error',20)
        
    def __call__(self,index): #check keys
        data = self.paramsets['data'][index]
        # compute convergence
        cs_error = list()
        cs_time = list()
        fail = 0
        for time1, error in zip(data['time'],data['error']):
            if True:
                if np.max(error[self.tmin:self.tmax]>self.thr_error): fail += 1
                else:
                    cs_error.append(error)
                    cs_time.append(time1)
            #except:
            #    eprint('Error processing {:s}'.format(file_name))
            #    return None

        if len(cs_error)>=1: fail = 100.*fail/len(cs_error)
        else: fail = 100
        
        # return [cs_error, cs_time, fail]

        # compute time series
        if 'rss_interval' in self.paramsets['keys']:
            j = self.paramsets['keys'].index('rss_interval')
            rss_interval = self.paramsets['comb'][index][j]
        else:
            rss_interval = 1
        
    
        time1  = np.hstack(cs_time)
        error = np.hstack(cs_error)

        #scaling
        min_time  = np.floor(np.min(time1))
        max_time  = np.floor(np.max(time1))+1
        time1     = time1-min_time
        max_time  = max_time-min_time

        #base time vector
        ts_time = range(0,int(max_time),rss_interval)

        #computining timeseries from data
        ts_error = list()
        ts_std    = list()

        for t in ts_time:
            x1 = [tserror for tt,tserror in zip(time1,error) if (tt>t)and(tt<t+rss_interval)]
            ts_error.append(np.mean(x1))
            ts_std.append(np.std(x1))

        ts_error  = np.asarray(ts_error)
        ts_std  = np.asarray(ts_std)
        
        #test['ts_min_time'] = min_time
        #test['ts_time']  = ts_time
        #test['ts_error'] = ts_error
        #test['ts_std']   = ts_std

        ts_metrics  = [np.max(ts_error[5:-5]),np.mean(ts_error[5:-5]),np.mean(ts_std[5:-5])]
        
        #return [min_time, ts_time, ts_error, ts_std, ts_metrics]
        
        #compute_cdf(paramsets,pindex,verbose=False):
        
        x = np.linspace(0,self.thr_error,1001)
        cdf_error  = np.zeros_like(x)
        cdf_std    = np.zeros_like(x)

        for j in range(len(x)): 
            _cdf_error = np.zeros(len(cs_error))
            for i, error in enumerate(cs_error):
                error = np.sort(error)
                errorsize = error.shape[0]
                try:
                    _cdf_error[i] = np.argwhere(error>x[j])[0]
                except:
                    _cdf_error[i] = errorsize
                _cdf_error[i] = _cdf_error[i]*1./errorsize
            cdf_error[j] = np.mean(_cdf_error)
            cdf_std[j] = np.std(_cdf_error)

        #closest points to .8, .9, .95
        i80 = np.argwhere(cdf_error>0.80)
        i90 = np.argwhere(cdf_error>0.90)
        i95 = np.argwhere(cdf_error>0.95)

        x80 = x[i80[0]-1] if (len(i80)!=0) else None
        x90 = x[i90[0]-1] if (len(i90)!=0) else None
        x95 = x[i95[0]-1] if (len(i95)!=0) else None

        # test['cdf_x'] = x
        # test['cdf_error'] = cdf_error
        # test['cdf_std']   = cdf_std
        # test['cdf_metrics'] = (float(x80),float(x90),float(x95))
        cdf_metrics = [float(x80),float(x90),float(x95)]
        # return [cdf_x, cdf_error, cdf_std, cdf_metrics]
        
        return [cs_error, cs_time, fail, min_time, ts_time, ts_error, ts_std, ts_metrics, x, cdf_error, cdf_std, cdf_metrics]

   
    def run(self,pool_workers=100):
        pool  = multiprocessing.Pool(pool_workers)
        
        for index,res in enumerate(pool.map(self,range(len(self.paramsets['data'])))):
            self.paramsets['data'][index]['cs_error']    = res[0]
            self.paramsets['data'][index]['cs_time']     = res[1]
            self.paramsets['data'][index]['cs_fail']     = res[2]
            self.paramsets['data'][index]['ts_min_time'] = res[3]
            self.paramsets['data'][index]['ts_time']     = res[4]
            self.paramsets['data'][index]['ts_error']    = res[5]
            self.paramsets['data'][index]['ts_std']      = res[6]
            self.paramsets['data'][index]['ts_metrics']  = res[7]
            self.paramsets['data'][index]['cdf_x']       = res[8]
            self.paramsets['data'][index]['cdf_error']   = res[9]
            self.paramsets['data'][index]['cdf_std']     = res[10]
            self.paramsets['data'][index]['cs_metrics']  = res[11]

        pool.close()
        pool.join()