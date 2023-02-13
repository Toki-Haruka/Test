###########################################import and input

import numpy as np
import pandas as pd
import os,re
import multiprocessing 
import h5py
import csv
import ujson
from operator import itemgetter
from collections import defaultdict
from io import StringIO
import math
import gzip
import numpy
import os
import pandas
from functools import reduce
from collections import defaultdict

#from . import helper
#from ..utils import misc

n_processes = 8
eventalign_filepath = "/histor/zhao/donghan/04.Toki/04.Pore-C/00.Alignment/00.Native/08.Minimap2_Eventalign_Collapse/Native.tsv"
chunk_size = 2000000
out_dir = "/histor/zhao/donghan/04.Toki/04.Pore-C/00.Alignment/00.Native/13.Minimap2_xpore/00.Native"
skip_eventalign_indexing = False

readcount_min = 15

###########################################utils.misc

def makedirs(main_dir, sub_dirs=None, opt='depth'):
    if not os.path.exists(main_dir):
        os.makedirs(main_dir)
    filepaths = dict()
    if sub_dirs is not None:
        if opt == 'depth':
            path = main_dir
            for sub_dir in sub_dirs:
                path = os.path.join(path, sub_dir)
                filepaths[sub_dir] = path
                # if not os.path.exists(path):
                try:  # Use try-catch for the case of multiprocessing.
                    os.makedirs(path)
                except:
                    pass

        else:  # opt == 'breadth'
            for sub_dir in sub_dirs:
                path = os.path.join(main_dir, sub_dir)
                filepaths[sub_dir] = path
                # if not os.path.exists(path):
                try:  # Use try-catch for the case of multiprocessing.
                    os.makedirs(path)
                except:
                    pass

    return filepaths


def str_decode(df):
    str_df = df.select_dtypes([np.object])
    str_df = str_df.stack().str.decode('utf-8').unstack()
    for col in str_df:
        df[col] = str_df[col]
    return df


def str_encode(df):
    str_df = df.select_dtypes([np.object])
    str_df = str_df.stack().str.encode('utf-8').unstack()
    for col in str_df:
        df[col] = str_df[col]
    return df

###########################################scripts/helper




class EventalignFile:
    
    def __init__(self, fn):
        self._fn = fn
        self._open()

    def _open(self):
        fn = self._fn
        if os.path.splitext(fn)[1] == '.gz':
            self._handle = gzip.open(fn)
            self._decode_method = bytes.decode
        else:
            self._handle = open(fn)
            self._decode_method = str

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close()

    def close(self):
        self._handle.close()

    def readline(self):
        self._handle.readline()

    def __iter__(self):
        return self

    def __next__(self):
        return self._decode_method(next(self._handle))


def decor_message(text,opt='simple'):
    text = text.upper()
    if opt == 'header':
        return text
    else:
        return '--- ' + text + ' ---\n'

def end_queue(task_queue,n_processes):
    for _ in range(n_processes):
        task_queue.put(None)
    return task_queue
        
def get_ids(f_index,data_info): #todo
    df_list = []
      
    for condition_name, run_names in data_info.items():

        list_of_set_ids = []
        for run_name in run_names:
            list_of_set_ids += [set(f_index[run_name].keys())]

        # ids = reduce(lambda x,y: x.intersection(y), list_of_set_ids)
        ids = reduce(lambda x,y: x.union(y), list_of_set_ids)

        df_list += [pandas.DataFrame({'ids':list(ids),condition_name:[1]*len(ids)})]

    df_merged = reduce(lambda  left,right: pandas.merge(left,right,on=['ids'], how='outer'), df_list).fillna(0).set_index('ids')

    return sorted(list(df_merged[df_merged.sum(axis=1) >= 2].index)) # At least two conditions.

class Consumer(multiprocessing.Process):
    """ For parallelisation """
    
    def __init__(self,task_queue,task_function,locks=None,result_queue=None):
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue

        self.locks = locks

        self.task_function = task_function

        self.result_queue = result_queue

    def run(self):

        proc_name = self.name
        while True:
            next_task_args = self.task_queue.get()

            if next_task_args is None:
                self.task_queue.task_done()

                break

            result = self.task_function(*next_task_args,self.locks)

            self.task_queue.task_done()

            if self.result_queue is not None:
                self.result_queue.put(result)

def read_last_line(filepath): # https://stackoverflow.com/questions/3346430/what-is-the-most-efficient-way-to-get-first-and-last-line-of-a-text-file/3346788
    if not os.path.exists(filepath):
        return
    with open(filepath, "rb") as f:
        first = f.readline()        # Read the first line.
        if first == b'':
            return
        f.seek(-2, os.SEEK_END)     # Jump to the second last byte.
        while f.read(1) != b"\n":   # Until EOL is found...
            f.seek(-2, os.SEEK_CUR) # ...jump back the read byte plus one more.
        last = f.readline()         # Read last line.
    return last

def is_successful(filepath):
    return read_last_line(filepath) == b'--- SUCCESSFULLY FINISHED ---\n'
    

###########################################index

def index(eventalign_result,pos_start,out_paths,locks):
    
    eventalign_result = eventalign_result.set_index(['contig','namae','read_index'])
    Hikari = 0
    pos_end=pos_start

    with locks['index'], open(out_paths['index'],'a') as f_index:
        #print(list(dict.fromkeys(eventalign_result.index)))
        for index in list(dict.fromkeys(eventalign_result.index)):
            
            Chrom_id,St_range,read_index = index
            pos_end += eventalign_result.loc[index]['line_length'].sum()
            #print(f"Hikari:{Hikari}\tChromID: {Chrom_id}\tStart Range: {St_range}\tRead Index:{read_index}\tStartP:{pos_start}\tEndP:{pos_end}\n")
            try: # sometimes read_index is nan
                f_index.write('%s\t%d\t%d\t%d\t%d\n' %(Chrom_id,St_range,read_index,pos_start,pos_end))
            except:
                #print("???")
                pass
            pos_start = pos_end

def nearest_100K(x):
    return int(math.ceil(x / 100000.0)) * 100000

def parallel_index(eventalign_filepath,chunk_size,out_dir,n_processes):
    
##### Create output paths and locks.
    
    out_paths,locks = dict(),dict()
    
    for out_filetype in ['index']:
        out_paths[out_filetype] = os.path.join(out_dir,'eventalign.%s' %out_filetype)
        locks[out_filetype] = multiprocessing.Lock()
        
##### Create empty files.

    with open(out_paths['index'],'w') as f:
        f.write('Chrom_id\tSt_range\tread_index\tpos_start\tpos_end\n') # header
        
##### Create communication queues.

    task_queue = multiprocessing.JoinableQueue(maxsize=n_processes * 2)

##### Create and start consumers.

    consumers = [Consumer(task_queue=task_queue,task_function=index,locks=locks) for i in range(n_processes)]#改回·删了helper
    
    for p in consumers:
        p.start()
        
##### Load tasks into task_queue. A task is eventalign information of one read.
    
    eventalign_file = open(eventalign_filepath,'r')
    pos_start = len(eventalign_file.readline()) #remove header
    chunk_split = None
    index_features = ['contig','read_index','line_length','namae','position']
    
    for chunk in pd.read_csv(eventalign_filepath, chunksize=chunk_size,sep='\t'):
        chunk_complete = chunk[chunk['read_index'] != chunk.iloc[-1]['read_index']]
        chunk_concat = pd.concat([chunk_split,chunk_complete])
        chunk_concat_size = len(chunk_concat.index)
        lines = [len(eventalign_file.readline()) for i in range(chunk_concat_size)]
        chunk_concat['line_length'] = np.array(lines)
        chunk_concat["namae"] = chunk_concat["position"].apply(nearest_100K)
        task_queue.put((chunk_concat[index_features],pos_start,out_paths))
        pos_start += sum(lines)
        chunk_split = chunk[chunk['read_index'] == chunk.iloc[-1]['read_index']]
        
    chunk_split_size = len(chunk_split.index)
    lines = [len(eventalign_file.readline()) for i in range(chunk_split_size)]
    chunk_split['line_length'] = np.array(lines)
    chunk_split["namae"] = chunk_split["position"].apply(nearest_100K)
    task_queue.put((chunk_split[index_features],pos_start,out_paths))
    task_queue = end_queue(task_queue,n_processes)
    task_queue.join()

########################################################pre_processing

def combine(events_str):
    f_string = StringIO(events_str)
    eventalign_result = pd.read_csv(f_string,delimiter='\t',names=['contig','position','reference_kmer','read_index',
                         'strand','event_index','event_level_mean','event_stdv','event_length','model_kmer',
                         'model_mean', 'model_stdv', 'standardized_level', 'start_idx', 'end_idx'])
    f_string.close()
    cond_successfully_eventaligned = eventalign_result['reference_kmer'] == eventalign_result['model_kmer']
    if cond_successfully_eventaligned.sum() != 0:

        eventalign_result = eventalign_result[cond_successfully_eventaligned]
        eventalign_result['namae'] = eventalign_result["position"].apply(nearest_100K)
        
        keys = ['read_index','contig','namae','position','reference_kmer'] # for groupby
        
        eventalign_result['length'] = pd.to_numeric(eventalign_result['end_idx'])-pd.to_numeric(eventalign_result['start_idx'])
        eventalign_result['sum_norm_mean'] = pd.to_numeric(eventalign_result['event_level_mean']) * eventalign_result['length']

        eventalign_result = eventalign_result.groupby(keys)

        sum_norm_mean = eventalign_result['sum_norm_mean'].sum() 
        start_idx = eventalign_result['start_idx'].min()
        end_idx = eventalign_result['end_idx'].max()
        total_length = eventalign_result['length'].sum()

        eventalign_result = pd.concat([start_idx,end_idx],axis=1)
        eventalign_result['norm_mean'] = (sum_norm_mean/total_length).round(1)
        eventalign_result.reset_index(inplace=True)

        eventalign_result['Chrom_id'] = [contig for contig in eventalign_result['contig']]

   #     eventalign_result['transcriptomic_position'] = pd.to_numeric(eventalign_result['position']) + 2 # the middle position of 5-mers.

        features = ['Chrom_id','namae','position','reference_kmer','norm_mean']
#        np_events = eventalign_result[features].reset_index().values.ravel().view(dtype=[('transcript_id', 'S15'), ('transcriptomic_position', '<i8'), ('reference_kmer', 'S5'), ('norm_mean', '<f8')])
        df_events = eventalign_result[features]
        np_events = np.rec.fromrecords(df_events, names=[*df_events])
        return np_events

def preprocess_tx(tx_id,data_dict,out_paths,locks):
    events = []
    
    if len(data_dict) == 0:
        return

    for read_id,events_per_read in data_dict.items():
        events += [events_per_read]
    
    events = np.concatenate(events)
    idx_sorted = np.argsort(events['position'])
    unique_positions, index = np.unique(events['position'][idx_sorted],return_index = True)
    
    y_arrays = np.split(events['norm_mean'][idx_sorted], index[1:])
    reference_kmer_arrays = np.split(events['reference_kmer'][idx_sorted], index[1:])
    
    data = defaultdict(dict)
    asserted = True
    
    for position,y_array,reference_kmer_array in zip(unique_positions,y_arrays,reference_kmer_arrays):
        position = int(position)
        
        if (len(set(reference_kmer_array)) == 1) and ('XXXXXX' in set(reference_kmer_array)) or (len(y_array) == 0):
            continue
            
        if 'XXXXXX' in set(reference_kmer_array):
            y_array = y_array[reference_kmer_array != 'XXXXXX']  
            assert len(y_array) == len(reference_kmer_array) - (reference_kmer_array=='XXXXXX').sum()
            reference_kmer_array = reference_kmer_array[reference_kmer_array != 'XXXXXX']
            
        try:
            assert len(set(reference_kmer_array)) == 1
            assert list(set(reference_kmer_array))[0].count('N') == 0 ##to weed out the mapped kmers from tx_seq that contain 'N', which is not in diffmod's model_kmer

        except:
            asserted = False
            break
            
        kmer = set(reference_kmer_array).pop()
        
        data[position] = {kmer: list(np.around(y_array,decimals=2))}
        
    log_str = '%s: %s.' %(str(tx_id),asserted)

    with locks['json'], open(out_paths['json'],'a') as f:
        pos_start = f.tell()
        f.write('{')
        f.write('"%s":' %str(tx_id))
        ujson.dump(data, f)
        f.write('}\n')
        pos_end = f.tell()
        
    with locks['index'], open(out_paths['index'],'a') as f:
        f.write('%s,%d,%d\n' %(str(tx_id),pos_start,pos_end))
        
    with locks['readcount'], open(out_paths['readcount'],'a') as f: #todo: repeats no. of tx >> don't want it.
        n_reads = len(data_dict)
        f.write('%s,%d\n' %(str(tx_id),n_reads))
        
    with locks['log'], open(out_paths['log'],'a') as f:
        f.write(log_str + '\n')

def parallel_preprocess_tx(eventalign_filepath,out_dir,n_processes,readcount_min):
    
##### Create output paths and locks.
    
    out_paths,locks = dict(),dict()
    
    for out_filetype in ['json','index','log','readcount']:
        out_paths[out_filetype] = os.path.join(out_dir,'data.%s' %out_filetype)
        locks[out_filetype] = multiprocessing.Lock()
        
##### Writing the starting of the files.#####################################################################要改#########################
    
    open(out_paths['json'],'w').close()
    
    with open(out_paths['index'],'w') as f:
        f.write('ChromID\tPosRange\tStart\tEnd\n') # header#改·加东西
        
    with open(out_paths['readcount'],'w') as f:
        f.write('ChromID\tPosRange\tn_reads\n') # header#改·加东西
        
    open(out_paths['log'],'w').close()
    
##### Create communication queues.
    
    task_queue = multiprocessing.JoinableQueue(maxsize=n_processes * 2)
    
    consumers = [Consumer(task_queue=task_queue,task_function=preprocess_tx,locks=locks) for i in range(n_processes)]
    
    for p in consumers:
        p.start()
    
##### Load tasks into task_queue.############################################################################要改#########################

    tx_ids_processed = []
    
    df_eventalign_index = pd.read_csv(os.path.join(out_dir,'eventalign.index'),sep="\t")

    #tx_ids = (df_eventalign_index['Chrom_id'].values,df_eventalign_index['St_range'].values).tolist()
    tx_ids = [(row.Chrom_id,row.St_range) for row in df_eventalign_index.itertuples(index=False)]
    #print(tx_ids)

    tx_ids = list(dict.fromkeys(tx_ids))
    df_eventalign_index.set_index(['Chrom_id','St_range'],inplace=True)
    with open(eventalign_filepath,'r') as eventalign_result:
        
        for tx_id in tx_ids:
            data_dict = dict()
            readcount = 0
            
            for _,row in df_eventalign_index.loc[[tx_id]].iterrows():
                read_index,pos_start,pos_end = row['read_index'],row['pos_start'],row['pos_end']
                eventalign_result.seek(pos_start,0)
                events_str = eventalign_result.read(pos_end-pos_start)
                data = combine(events_str)
                
                if (data is not None) and (data.size > 1):
                    data_dict[read_index] = data
                    
                readcount += 1
                
            if readcount>=readcount_min:
                task_queue.put((tx_id,data_dict,out_paths)) # Blocked if necessary until a free slot is available. 
                tx_ids_processed += [tx_id]
                
    task_queue = end_queue(task_queue,n_processes)
    
    task_queue.join()
    
    with open(out_paths['log'],'a+') as f:
        f.write('Total %d Namaes.\n' %len(tx_ids_processed))
        f.write(decor_message('successfully finished'))

################################################Main

parallel_index(eventalign_filepath,chunk_size,out_dir,n_processes)

parallel_preprocess_tx(eventalign_filepath,out_dir,n_processes,readcount_min)


