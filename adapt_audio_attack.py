"""
    Author: Moustafa Alzantot (malzantot@ucla.edu)
    All rights reserved.
"""
import os, sys
import numpy as np
import sys
import time
import tensorflow as tf
from speech_commands import label_wav
import pydub
from scipy.io import wavfile
from multiprocessing import Process, Lock
from multiprocessing.sharedctypes import Value, Array
from multiprocessing.dummy import Pool
import subprocess
import argparse

#./run_adapt_attack1.sh 15 1000 20


def softmax(x):
#  return (x)
  e_x = np.exp(x - np.max(x))
  return e_x / e_x.sum()

def speedx(snd_array, factor):
    """ Speeds up / slows down a sound, by some factor. """
    indices = np.round(np.arange(0, len(snd_array), factor))
    indices = indices[indices < len(snd_array)].astype(int)
    return snd_array[indices]

def save_audiofile(output, filename):
    with open(filename, 'wb') as fh:
        fh.write(output)

def load_audiofile(filename):
    with open(filename, 'rb') as fh:
        return fh.read()

def load_graph(filename):
    with tf.gfile.FastGFile(filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

def load_labels(filename):
    return [line.rstrip() for line in tf.gfile.FastGFile(filename)]
        

def print_output(output_preds, labels):
    top_k = output_pred.argsort()[-5:][::-1]
    for node_id in top_k:
        human_string = labels[node_id]
        score = output_pred[node_id]
        print('%s %d score = %0.5f' %(human_string, node_id, score))
    print('----------------------')

################## GenAttack again ?
# TODO(malzantot): any thoughts about byte ordering ?
header_len = 44
data_max = 32767
data_min = -32768
mutation_p = 0.0005 #0.0005

def gen_population_member(x_orig, eps_limit):
    new_bytearray = bytearray(x_orig)
    # step = 2
    # if bps == 8:
    step = 2
    for i in range(header_len, len(x_orig), step):
        if np.random.random() < mutation_p:
        #    if np.random.random() < 0.5:
        #        new_bytearray[i] = min(255, new_bytearray[i]+1)
        #    else:
        #        new_bytearray[i] = max(0, new_bytearray[i]-1)
            int_x = int.from_bytes(x_orig[i:i+2], byteorder='little', signed=True)
            new_int_x = min(data_max, max(data_min, int_x + np.random.choice(range(-eps_limit, eps_limit))))
            new_bytes = int(new_int_x).to_bytes(2, byteorder='little', signed=True)
            new_bytearray[i] = new_bytes[0]
            new_bytearray[i+1] = new_bytes[1]
    return bytes(new_bytearray)

def crossover(x1, x2):
    ba1 = bytearray(x1)
    ba2 = bytearray(x2)
    step = 2
    # if bps == 8:
    #    step = 1
    for i in range(header_len, len(x1), step):
        if np.random.random() < 0.5:
            ba2[i] = ba1[i]
    return bytes(ba2)

# def refine(x_new, x_orig, pbs=16, limit=10):
#    ba_new = bytearray(x_new)
#    ba_orig = bytearray(x_orig)
#    step = 2
#    if pbs == 8:
#        step = 1
#    for i in range(header_len, len(x_new), step):
#        # if np.random.random() < 0.5:
#        ba_new[i] = min(ba_orig[i]+limit, max(ba_orig[i]-limit, ba_new[i]))
#        ba_new[i] = min(255, max(0, ba_new[i]))
#    return bytes(ba_new)

def mutation(x, eps_limit):
    ba = bytearray(x)
    step = 2
    #if pbs == 8:
    #    step = 1
    for i in range(header_len, len(x), step):
        #if np.random.random() < 0.05:
        # ba[i] = max(0, min(255, np.random.choice(list(range(ba[i]-4, ba[i]+4)))))
        #elif np.random.random() < 0.10:
        #ba[i] = max(0, min(255, ba[i] + np.random.choice([-1, 1])))
        if np.random.random() < mutation_p:
            int_x = int.from_bytes(ba[i:i+2], byteorder='big', signed=True)
            new_int_x = min(data_max, max(data_min, int_x + np.random.choice(range(-eps_limit, eps_limit))))
            new_bytes = int(new_int_x).to_bytes(2, byteorder='big', signed=True)
            ba[i] = new_bytes[0]
            ba[i+1] = new_bytes[1]
    return bytes(ba)


def predict(sess, output_tensor, input_tensor, filename, rando, index):
    if(index == 0):
        wav_data = load_audiofile(filename)
        preds = sess.run(output_tensor, feed_dict = {
                    input_tensor: wav_data
                    })
        return(preds[0])
    if(index == 1):
        array = pydub.audio_segment.AudioSegment.from_wav(filename)
        newarray = pydub.effects.high_pass_filter(array, 40)
        newarray = pydub.effects.low_pass_filter(newarray, 8000)
        newarray.export("%dbpf.wav" %(rando), format="wav")
        wav_data = load_audiofile("%dbpf.wav" %(rando))
        preds = sess.run(output_tensor, feed_dict = {
                    input_tensor: wav_data
                    })
        return(preds[0])
    if(index == 2):
        fps, array = wavfile.read(filename)
        array = np.copy(array)
        array = speedx(array,0.99)
        wavfile.write("%dpan.wav" %(rando), fps, (1*array).astype('int16'))
        array = pydub.audio_segment.AudioSegment.from_wav("%dpan.wav" %(rando))
        newarray = pydub.effects.pan(array, 0.4)
        newarray.export("%dpan.wav" %(rando), format="wav")
        wav_data = load_audiofile("%dpan.wav" %(rando))
        preds = sess.run(output_tensor, feed_dict = {
                    input_tensor: wav_data
                    })
        return(preds[0])
    return(0)

def ensemble_score(sess, x, target, input_tensor, output_tensor, rando):
    #return(score(sess, x, target, input_tensor, output_tensor))
    #print("Begin Score")
   # rando = np.random.randint(sys.maxsize)
    save_audiofile(x, "%d.wav" %(rando))
    filename = "%d.wav" %(rando)
    subprocess.call(format("wine ../speexwin32/bin/speexenc.exe %s %d.spx --vad --narrowband --comp 8 --quality 5 2> tmp.txt; wine ../speexwin32/bin/speexdec.exe %d.spx %d.wav 2> tmp.txt" %(filename, rando, rando, rando)), shell=True)
  #  wav_data = load_audiofile(filename)
  #  preds = sess.run(output_tensor, feed_dict = {
  #                  input_tensor: wav_data
  #                  })
  #  nothing = preds

  #  array = pydub.audio_segment.AudioSegment.from_wav(filename)
  #  newarray = pydub.effects.high_pass_filter(array, 40)
  #  newarray = pydub.effects.low_pass_filter(newarray, 8000)
  #  newarray.export("%dbpf.wav" %(rando), format="wav")
  #  wav_data = load_audiofile("%dbpf.wav" %(rando))
  #  preds = sess.run(output_tensor, feed_dict = {
  #                  input_tensor: wav_data
  #                  })
  #  lpf = preds

  #  fps, array = wavfile.read(filename)
  #  array = np.copy(array)
  #  array = speedx(array,0.99)
  #  wavfile.write("%dpan.wav" %(rando), fps, (1*array).astype('int16'))
  #  array = pydub.audio_segment.AudioSegment.from_wav("%dpan.wav" %(rando))
  #  newarray = pydub.effects.pan(array, 0.4)
  #  newarray.export("%dpan.wav" %(rando), format="wav")
  #  wav_data = load_audiofile("%dpan.wav" %(rando))
  #  preds = sess.run(output_tensor, feed_dict = {
  #                  input_tensor: wav_data
  #                  })
  #  pan = preds
 
    #os.system("rm %d.wav; rm %d.spx; rm %dbpf.wav; rm %dpan.wav" %(rando, rando, rando, rando))
   # pool = Pool(3)
   # logits = pool.starmap(predict, zip([sess, sess, sess], [output_tensor, output_tensor, output_tensor], [input_tensor, input_tensor, input_tensor], [filename, filename, filename], [rando, rando, rando], [0,1,2]))
   # pool.close()
   # pool.join()
    logits = [predict(sess, output_tensor, input_tensor, filename, rando,i) for i in range(3)]
    ret = softmax(np.asarray(logits[0])) + softmax(np.asarray(logits[1])) + softmax(np.asarray(logits[2]))
    #print("End Score")
    return(ret/3)
    

def score(sess, x, target, input_tensor, output_tensor):
    output_preds, = sess.run(output_tensor,
        feed_dict={input_tensor: x})
    return output_preds

def threading_ensemble_score(sess, x, target, input_tensor, output_tensor, ts, tp, mp, mpred, index, rando):
    vector = ensemble_score(sess, x, target, input_tensor, output_tensor, rando)
   # vector = score(sess, x, target, input_tensor, output_tensor)
    tscore = vector[target] - np.max(vector)
    ts[index] = tscore
    tp[index] = vector[target]
    mp[index] = np.max(vector)
    mpred[index] = np.argmax(vector)
    return(1)


key = ["silence", "background", "yes", "no", "up", "down", "left", "right", "on",
       "off", "stop", "go"]

def generate_attack(x_orig, target, limit, sess, input_node,
    output_node, max_iters, eps_limit, verbose, index):
    print("yo wtf")
    pop_size = 500 # 20
    elite_size = 2 #2
    temp = 0.01 # 0.01
    initial_pop = [gen_population_member(x_orig, eps_limit) for _ in range(pop_size)]
    print("ip")
    randos = [np.random.randint(sys.maxsize) for _ in range(pop_size)]
    print("randos")
    for idx in range(max_iters):
        print("Example %d, Iteration %d    " %(index, idx))#, end="\r")
        #pop_scores = np.array([ensemble_score(sess, x, target, input_node, output_node) for x in initial_pop])
        #target_scores = pop_scores[:, target] - np.max(pop_scores, axis=1) # before it was just pop_scores[:, target]
        
        pool = Pool(pop_size)

        ts = Array('d', [0 for _ in range(pop_size)])
        tp = Array('d', [0 for _ in range(pop_size)])
        mp = Array('d', [0 for _ in range(pop_size)])
        mpred = Array('d', [0 for _ in range(pop_size)])

        ps = []
        #randos = [np.random.randint(sys.maxsize) for _ in range(pop_size)]
        #for i in range(pop_size):
        #    p = Process(target=speex, args=(initial_pop[i], randos[i]))
        #    p.start()
        #    ps.append(p)

        #for p in ps:
        #    p.join()   
      
        kappa = pool.starmap(threading_ensemble_score, zip([sess for _ in range(pop_size)], initial_pop[:pop_size], [target for _ in range(pop_size)], [input_node for _ in range(pop_size)], [output_node for _ in range(pop_size)], [ts for _ in range(pop_size)], [tp for _ in range(pop_size)], [mp for _ in range(pop_size)], [mpred for _ in range(pop_size)], [i for i in range(pop_size)], randos))
# [(sess, initial_pop[i], target, input_node, output_node, ts, tp, mp, mpred, i) for i in range(pop_size)])
        pool.close()
        pool.join()

        target_scores = np.asarray([k for k in ts])

        pop_ranks = list(reversed(np.argsort(target_scores)))
        elite_set = [initial_pop[x] for x in pop_ranks[:elite_size]]
        
        top_attack = initial_pop[pop_ranks[0]]
        #top_pred = np.argmax(pop_scores[pop_ranks[0],:])
      
        top_pred = mpred[pop_ranks[0]]

        print("Top Pred:", key[int(top_pred)], "Target Pred:", key[int(target)])
        #print("Top Score:", pop_scores[pop_ranks[0],target], "against", pop_scores[pop_ranks[0],top_pred])
        print("Top Score:", tp[pop_ranks[0]], "against", mp[pop_ranks[0]])
        #if verbose or not verbose:
        #    if top_pred == target:
        if top_pred == target:
            #os.system("rm *.wav; rm *.spx")
            print("*** SUCCESS ****, attack finished in %d iters" %idx)
            return top_attack, idx+1

        scores_logits = np.exp(target_scores /temp)
        pop_probs = scores_logits / np.sum(scores_logits)
        child_set = [crossover(
            initial_pop[np.random.choice(pop_size, p=pop_probs)],
            initial_pop[np.random.choice(pop_size, p=pop_probs)])
            for _ in range(pop_size - elite_size)]
        initial_pop = elite_set + [mutation(child, eps_limit) for child in child_set]
    print("*** FAILURE ***")
    #system.os("rm *.wav; rm *.spx")
    return top_attack, 501
        
#flags = tf.flags
#flags.DEFINE_string("data_dir", "", "Data dir")
#flags.DEFINE_string("output_dir", "", "Data dir")
#flags.DEFINE_string("target_label", "", "Target classification label")
#flags.DEFINE_integer("limit", 4, "Noise limit")
#flags.DEFINE_string("graph_path", "", "Path to frozen graph file.")
#flags.DEFINE_string("labels_path", "", "Path to labels file.")
#flags.DEFINE_boolean("verbose", False, "")
#flags.DEFINE_integer("max_iters", 200, "Maxmimum number of iterations")
#FLAGS = flags.FLAGS

#if __name__ == '__main__':
#    data_dir = FLAGS.data_dir
#    output_dir = FLAGS.output_dir
#    target_label = FLAGS.target_label
#    eps_limit = FLAGS.limit
#    graph_path = FLAGS.graph_path
#    labels_path = FLAGS.labels_path
#    max_iters = FLAGS.max_iters
#    verbose = FLAGS.verbose
def produce_example(sess, attack_target, source, i, index):
    print("Entering thread", index)
    output_dir = "../adapt_output1/result/" + key[attack_target] + "/" + key[source]
    data_dir = "../adapt_output1/data/" + key[source]
   
    input_node_name = 'wav_data:0'
    output_node_name = 'labels_softmax:0'
    eps_limit = 15
    graph_path = "ckpts/conv_actions_frozen.pb"
    labels_path = "ckpts/conv_actions_labels.txt"
    max_iters = 1000
    verbose =False

    labels = load_labels(labels_path)

    wav_files_list =\
        [f for f in os.listdir(data_dir) if f.endswith(".wav")]
    
    #target_idx = [idx for idx in range(len(labels)) if labels[idx]==target_label]
    #if len(target_idx) == 0:
    #    print("Target label not found.")
    #    sys.exit(1)
    #target_idx = target_idx[0]
    target_idx = attack_target


    #load_graph(graph_path)
    #config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True
   # with tf.Session(config=config) as sess:
    output_node = sess.graph.get_tensor_by_name(output_node_name) 
    for input_file in wav_files_list[i:][:1]:
        start_time = time.time()
        x_orig = load_audiofile(data_dir+'/'+input_file)
        #TODO(malzantot) fix
        # x_pbs = 1
        num_channels = int(x_orig[22]) + int(x_orig[23]*256)
        sample_rate = int(x_orig[24]) + int(x_orig[25]*256) + int(x_orig[26]*2**16) + int(x_orig[27]*2**24)
        pbs = int(x_orig[34])
        byte_rate = int(x_orig[28]) + int(x_orig[29]*256) + int(x_orig[30]*2**16) + int(x_orig[31]*2**24)
        chunk_id = chr(int(x_orig[0])) + chr(int(x_orig[1])) + chr(int(x_orig[2])) + chr(int(x_orig[3]))
        # if chunk_id == 'RIFF':
        #    # chunk_id='RIFF' -> little endian data form. 'RIFX'-> big endian form.
        #    header_len += 1
        assert chunk_id == 'RIFF', 'ONLY RIIF format is supported'

        if verbose:
            print("chunk id = %s" %chunk_id)
            print("bps = %d - num channels = %d - Sample rate = %d ." 
            %(pbs, num_channels, sample_rate))
            print("byte rate = %d" %(byte_rate))

        assert pbs == 16, "Only PBS=16 is supported now"
        print("rto") 
        attack_output = generate_attack(x_orig, target_idx, eps_limit,
            sess, input_node_name, output_node, max_iters, pbs, verbose, index)
        save_audiofile(attack_output[0], output_dir+'/'+input_file)
        print("Example saved to " + output_dir+'/'+input_file)
        end_time = time.time()
        return(attack_output[1], end_time-start_time)
        print("Attack done (%d iterations) in %0.4f seconds" %(max_iters, (end_time-start_time)))
        while(True):
            time.sleep(100)
                
       

parser = argparse.ArgumentParser(description="Creates an adaptive adversarial example.")
parser.add_argument('-i','--index', help="Index of the adversarial example to be created.", required=True)
args = vars(parser.parse_args())
argindex = int(args['index'])
fayal = open("adapt_log%d.txt" %argindex, "a")

graph_path = "ckpts/conv_actions_frozen.pb"
load_graph(graph_path)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    nexamp = 10
    trgts = []
    srcs = []
    indices = []
    for trgt in range(2,12):
        for src in range(2,12):
            if(src == trgt):
                continue
            for i in range(nexamp):
                trgts.append(trgt)
                srcs.append(src)
                indices.append(i)

    simulsize = 10
    for i in range(argindex,len(indices),simulsize):
        smalltrgts = trgts[i:i+simulsize]
        smallsrcs = srcs[i:i+simulsize]
        smallindices = indices[i:i+simulsize]
       # pool = Pool(simulsize)
       # times = pool.starmap(produce_example, zip([sess for _ in range(simulsize)], trgts, srcs, indices, range(simulsize)))
       # pool.close()
       # pool.join()
        times = produce_example(sess, trgts[argindex], srcs[argindex], indices[argindex], i)
        fayal.write("\nAttack no. " + str(argindex) + " successful. " + str(times[0]) + " iterations in " + str(times[1]) + "seconds.")
        print("\nAttack no. " + str(argindex) + " successful. " + str(times[0]) + " iterations in " + str(times[1]) + "seconds.")
        #successes = [k[0] for k in times if k[0] <= 500]
        #print("Batch " + str((i/simulsize) + 1) + ": " + str(len(successes)) + "out of " + str(simulsize))
