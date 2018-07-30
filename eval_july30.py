print("Starting session...")

import subprocess
import progressbar
import numpy as np
import tensorflow as tf
from speech_commands import label_wav
import os, sys
import csv
from scipy.io import wavfile
import wave
from numpy import *
import pandas as pd
import pydub
#from tqdm import tqdm
import keras.backend as K
from sklearn import tree
from sklearn.metrics.pairwise import manhattan_distances as l1
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from scipy.spatial import distance
from random import shuffle
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier

random.seed(10)

saving = False

rf = 0

passive_aggressive = 1

numtrees = 2

detection = 1

ensemble = 0 # 2 for ensemble in ICMLA paper

adversarial = 1

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

preprocessing = 13
# 855
# TODO: Try experimenting with negative speedx in combination with preprocessing 4. 

# 2 yes
# 3 no
# 4 up
# 5 down
# 6 left
# 7 right
# 8 on
# 9 off
# 10 stop
# 11 go

key = ["silence", "background", "yes", "no", "up", "down", "left", "right", "on",
       "off", "stop", "go"]

def softmax(x):
#  return (x)
  e_x = np.exp(x - np.max(x))
  return e_x / e_x.sum()

def L1(v1,v2):
      if(len(v1)!=len(v2)):
        print("error")
        return(-1)
      return(sum([abs(v1[i]-v2[i]) for i in range(len(v1))]))


def load_wav(filename):
    #try:
    wavedata=wavfile.read(filename)
    samplerate=int(wavedata[0])
    smp=wavedata[1]*(1.0/32768.0)
    if len(smp.shape)>1: #convert to mono
        smp=(smp[:,0]+smp[:,1])*0.5
    return (samplerate,smp)
    #except:
    #    print ("Error loading wav: "+filename)
    #    return None

def paulstretch(samplerate,smp,stretch,windowsize_seconds,outfilename):
    outfile=wave.open(outfilename,"wb")
    outfile.setsampwidth(2)
    outfile.setframerate(samplerate)
    outfile.setnchannels(1)

    #make sure that windowsize is even and larger than 16
    windowsize=int(windowsize_seconds*samplerate)
    if windowsize<16:
        windowsize=16
    windowsize=int(windowsize/2)*2
    half_windowsize=int(windowsize/2)

    #correct the end of the smp
    end_size=int(samplerate*0.05)
    if end_size<16:
        end_size=16
    smp[len(smp)-end_size:len(smp)]*=linspace(1,0,end_size)


    #compute the displacement inside the input file
    start_pos=0.0
    displace_pos=(windowsize*0.5)/stretch

    #create Hann window
    window=0.5-cos(arange(windowsize,dtype='float')*2.0*pi/(windowsize-1))*0.5

    old_windowed_buf=zeros(windowsize)
    hinv_sqrt2=(1+sqrt(0.5))*0.5
    hinv_buf=hinv_sqrt2-(1.0-hinv_sqrt2)*cos(arange(half_windowsize,dtype='float')*2.0*pi/half_windowsize)

    while True:

        #get the windowed buffer
        istart_pos=int(floor(start_pos))
        buf=smp[istart_pos:istart_pos+windowsize]
        if len(buf)<windowsize:
            buf=append(buf,zeros(windowsize-len(buf)))
        buf=buf*window

        #get the amplitudes of the frequency components and discard the phases
        freqs=abs(fft.rfft(buf))

        #randomize the phases by multiplication with a random complex number with modulus=1
        ph=random.uniform(0,2*pi,len(freqs))*1j
        freqs=freqs*exp(ph)

        #do the inverse FFT 
        buf=fft.irfft(freqs)

        #window again the output buffer
        buf*=window


        #overlap-add the output
        output=buf[0:half_windowsize]+old_windowed_buf[half_windowsize:windowsize]
        old_windowed_buf=buf

        #remove the resulted amplitude modulation
        output*=hinv_buf

        #clamp the values to -1..1 
        output[output>1.0]=1.0
        output[output<-1.0]=-1.0

        #write the output to wav file
        outfile.writeframes(int16(output*32767.0).tostring())

        start_pos+=displace_pos
        if start_pos>=len(smp):
            #print ("100 %")
            break
        #sys.stdout.write ("%d %% \r" % int(100.0*start_pos/len(smp)))
        sys.stdout.flush()

    outfile.close()


def stretch(snd_array, factor, window_size, h):
    """ Stretches/shortens a sound, by some factor. """
    phase = np.zeros(window_size)
    hanning_window = np.hanning(window_size)
    result = np.zeros(int(len(snd_array) / factor + window_size))

    for i in np.arange(0, len(snd_array) - (window_size + h), h*factor):
        i = int(i)
        # Two potentially overlapping subarrays
        a1 = snd_array[i: i + window_size]
        a2 = snd_array[i + h: i + window_size + h]

        # The spectra of these arrays
        s1 = np.fft.fft(hanning_window * a1)
        s2 = np.fft.fft(hanning_window * a2)

        # Rephase all frequencies
        phase = (phase + np.angle(s2/s1)) % 2*np.pi

        a2_rephased = np.fft.ifft(np.abs(s2)*np.exp(1j*phase))
        i2 = int(i/factor)
        result[i2: i2 + window_size] += hanning_window*a2_rephased.real

    # normalize (16bit)
    result = ((2**(16-4)) * result/result.max())

    return result.astype('int16')

def speedx(snd_array, factor):
    """ Speeds up / slows down a sound, by some factor. """
    indices = np.round(np.arange(0, len(snd_array), factor))
    indices = indices[indices < len(snd_array)].astype(int)
    return snd_array[indices]


def load_graph(filename):
    with tf.gfile.FastGFile(filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

def load_audiofile(filename):
    with open(filename, 'rb') as fh:
        return fh.read()

def pitchshift(snd_array, n, window_size=2**13, h=2**11):
    """ Changes the pitch of a sound by ``n`` semitones. """
    factor = 2**(1.0 * n / 12.0)
    stretched = stretch(snd_array, 1.0/factor, window_size, h)
    return speedx(stretched[window_size:], factor)

def predrandom2(filename, iters, sess):
    fps, array = wavfile.read(filename)
    array = np.copy(array)
    array = speedx(array,0.99)
    logits = np.zeros(12)
    for i in range(iters):
        newarray = np.asarray(np.copy(array))
        rands = np.random.uniform(0,1,(len(array)))
        wavfile.write("temp.wav",fps,(rands*newarray).astype('int16'))
        sound = load_audiofile("temp.wav")
        preds = sess.run(output_node, feed_dict = {
                    'wav_data:0': sound
                    })
        logits[np.argmax(preds[0])] += np.max(preds[0])
    return(np.argmax(logits))


def random_drop(array, prob):
    newarray = np.asarray(np.copy(array))
    rands = np.random.choice([0,1], size=(len(array)), p=[prob, 1-prob])
    return((newarray*rands).astype('int16'))

def predrandom(filename, prob, iters, sess):
    fps, array = wavfile.read(filename)
    array = np.copy(array)
    array = speedx(array,0.99)
    logits = np.zeros(12)
    for i in range(iters):
        newarray = np.asarray(np.copy(array))
        rands = np.random.choice([0.5, 1], size=(len(array)), p=[prob, 1-prob]) 
        wavfile.write("temp.wav",fps,(rands*newarray).astype('int16'))
        sound = load_audiofile("temp.wav")
        preds = sess.run(output_node, feed_dict = {
                    'wav_data:0': sound
                    })   
        logits[np.argmax(preds[0])] += np.max(preds[0])
    return(np.argmax(logits))

def decision_threshold(x, y):
    """ find point that has maxium information gain
        :Params : x: 2Darray of feature
        :Params : y: array of target
        :Return : int : of best x value"""
    
    model = DecisionTreeClassifier(max_depth=1, criterion='entropy')
    model.fit(x,y)
    print ("-- Uncertainty Threshold: ", model.tree_.threshold[0])
    return model.tree_.threshold[0]

def calculate_threshold(output_node, sess):
    print("Calculating Uncertainty Threshold")
    sound = load_audiofile("../output/data/left/3d3ddaf8_nohash_1.wav")
    times = [];
    for i in range(100):
        preds = sess.run(output_node, feed_dict = {
                         'wav_data:0': sound
                         })
        times.append(preds[0])
    print("Array of Variances:", np.var(times, axis=0))
    print("Mean Variance:", np.mean(np.var(times, axis=0)))
    print("Prediction Variance:", np.var(np.argmax(times, axis=1))) # Iffy
    print("Variance within Prediction:", np.var(times, axis=0)[np.argmax(np.mean(times, axis=0))])
    print("Prediction:", key[np.argmax(np.mean(times, axis=0))])


def opus_all():
    bar = progressbar.ProgressBar(max_value=1800)
    counter = 0
    for src in range(2,12):
        for trgt in range(2,12): #change back to 12 when complete.
            case_dir = format("../opus_out/result/%s/%s" %(key[trgt], key[src]))
            if os.path.exists(case_dir):
                wav_files =[format('%s/%s' %(case_dir, f)) for f in os.listdir(case_dir) if f.endswith('.wav')]
                for filename in wav_files:
                    os.system(format("wine ../opuswin64/opusenc.exe  %s temp.opus --quiet --bitrate 4 --expect-loss 30 --framesize 10; wine ../opuswin64/opusdec.exe temp.opus %s --quiet" %(filename,filename)))
                    counter+=1; bar.update(counter)

#print("Doing the opus.")
#opus_all()
#print("finished.")
#exit()

def ensemble2(filename, sess, output_node):
    os.system(format("wine ../speexwin32/bin/speexenc.exe %s temp.spx --vad --narrowband --comp 8 --quality 5; wine ../speexwin32/bin/speexdec.exe temp.spx speex.wav" %(filename)))
    filen = "temp.wav"

    wav_data = load_audiofile("speex.wav")
    preds = sess.run(output_node, feed_dict = {
                    'wav_data:0': wav_data
                    })
    nothing = preds[0]

    array = pydub.audio_segment.AudioSegment.from_wav("speex.wav")
    newarray = pydub.effects.high_pass_filter(array, 40)
    newarray = pydub.effects.low_pass_filter(newarray, 8000)
    newarray.export("temp.wav", format="wav")
    wav_data = load_audiofile(filen)
    preds = sess.run(output_node, feed_dict = {
                    'wav_data:0': wav_data
                    })
    lpf = preds[0]

    fps, array = wavfile.read("speex.wav")
    array = np.copy(array)
    array = speedx(array,0.99)
    wavfile.write("temp.wav", fps, (1*array).astype('int16'))
    array = pydub.audio_segment.AudioSegment.from_wav("temp.wav")
    newarray = pydub.effects.pan(array, 0.4)
    newarray.export("temp.wav", format="wav")
    filen = "temp.wav"
    wav_data = load_audiofile(filen)
    preds = sess.run(output_node, feed_dict = {
                    'wav_data:0': wav_data
                    })
    pan = preds[0]

    return((softmax(nothing),softmax(lpf),softmax(pan)))

def ensemble3(filename, sess, output_node):
     wav_data = load_audiofile(filename)
     preds = sess.run(output_node, feed_dict = {
                      'wav_data:0': wav_data
                      })
     nothing = preds[0]
     
     os.system(format("wine ../speexwin32/bin/speexenc.exe %s temp.spx --vad --narrowband --comp 8 --quality 5; wine ../speexwin32/bin/speexdec.exe temp.spx temp.wav" %(filename))) 
     wav_data = load_audiofile("temp.wav") # use "speex.wav" with nothing and nothing as the two other params for 98% adv acc
     preds = sess.run(output_node, feed_dict = {
                     'wav_data:0': wav_data
                     })
     speex = preds[0]
 
    # array = pydub.audio_segment.AudioSegment.from_wav(filename)
    # newarray = pydub.effects.high_pass_filter(array, 40)
    # newarray = pydub.effects.low_pass_filter(newarray, 8000)
    # newarray.export("temp.wav", format="wav")
    # wav_data = load_audiofile("temp.wav")
    # preds = sess.run(output_node, feed_dict = {
    #                 'wav_data:0': wav_data
    #                 })
    # lpf = preds[0]
 
     # fps, array = wavfile.read(filename)
     # array = np.copy(array)
     # array = speedx(array,0.99)
     # wavfile.write("temp.wav", fps, (1*array).astype('int16'))
     # array = pydub.audio_segment.AudioSegment.from_wav("temp.wav")
     # newarray = pydub.effects.pan(array, 0.4)
     # newarray.export("temp.wav", format="wav")
     # filen = "temp.wav"
     # wav_data = load_audiofile(filen) #should be filen
     # preds = sess.run(output_node, feed_dict = {
     #                'wav_data:0': wav_data
     #                })
     pan = nothing#preds[0]
 
     return((nothing, speex, pan)) # should softmax

def ensemble4(filename, sess, output_node): #nothing & speex and nothing & pan
     wav_data = load_audiofile(filename)
     preds = sess.run(output_node, feed_dict = {
                      'wav_data:0': wav_data
                      })
     nothing = preds[0]

     os.system(format("wine ../speexwin32/bin/speexenc.exe %s temp.spx --vad --narrowband --comp 8 --quality 5; wine ../speexwin32/bin/speexdec.exe temp.spx temp.wav" %(filename)))
     wav_data = load_audiofile("temp.wav") # use "speex.wav" with nothing and nothing as the two other params for 98% adv acc
     preds = sess.run(output_node, feed_dict = {
                     'wav_data:0': wav_data
                     })
     speex = preds[0]

     fps, array = wavfile.read(filename)
     array = np.copy(array)
     array = speedx(array,0.99)
     wavfile.write("temp.wav", fps, (1*array).astype('int16'))
     array = pydub.audio_segment.AudioSegment.from_wav("temp.wav")
     newarray = pydub.effects.pan(array, 0.4)
     newarray.export("temp.wav", format="wav")
     filen = "temp.wav"
     wav_data = load_audiofile(filen) #should be filen
     preds = sess.run(output_node, feed_dict = {
                    'wav_data:0': wav_data
                    })
     pan = preds[0]
     return((nothing, speex, pan))  

def get_logits(index, sess, output_node, advben, prep, tdir):
     path = format("%s/%s/%s/%d.wav" %(tdir, prep, advben, index))
     wav_data = load_audiofile(path) #should be filen
     preds = sess.run(output_node, feed_dict = {
                    'wav_data:0': wav_data
                    })
     return(preds[0])

def fast_ensemble4(index, sess, output_node, advben, tdir):
     nothing = get_logits(index, sess, output_node, advben, "raw", tdir)
     speex = get_logits(index, sess, output_node, advben, "speex", tdir)
     pan = get_logits(index, sess, output_node, advben, "pan", tdir)
     return((nothing, speex, pan))

def ensemble5(filename, sess, output_node): #nothing & speex and nothing & BPF
     wav_data = load_audiofile(filename)
     preds = sess.run(output_node, feed_dict = {
                      'wav_data:0': wav_data
                      })
     nothing = preds[0]

     os.system(format("wine ../speexwin32/bin/speexenc.exe %s temp.spx --vad --narrowband --comp 8 --quality 5; wine ../speexwin32/bin/speexdec.exe temp.spx temp.wav" %(filename)))
     wav_data = load_audiofile("temp.wav") # use "speex.wav" with nothing and nothing as the two other params for 98% adv acc
     preds = sess.run(output_node, feed_dict = {
                     'wav_data:0': wav_data
                     })
     speex = preds[0]

     array = pydub.audio_segment.AudioSegment.from_wav(filename)
     newarray = pydub.effects.high_pass_filter(array, 40)
     newarray = pydub.effects.low_pass_filter(newarray, 8000)
     newarray.export("temp.wav", format="wav")
     wav_data = load_audiofile("temp.wav")
     preds = sess.run(output_node, feed_dict = {
                     'wav_data:0': wav_data
                     })
     lpf = preds[0]

     return((nothing, speex, lpf))

def fast_ensemble5(index, sess, output_node, advben, tdir):
     nothing = get_logits(index, sess, output_node, advben, "raw", tdir)
     speex = get_logits(index, sess, output_node, advben, "speex", tdir)
     lpf = get_logits(index, sess, output_node, advben, "bpf", tdir)
     return((nothing,  lpf, speex))

def ensemble6(filename, sess, output_node): #nothing & pan and nothing & BPF
     wav_data = load_audiofile(filename)
     preds = sess.run(output_node, feed_dict = {
                      'wav_data:0': wav_data
                      })
     nothing = preds[0]

     fps, array = wavfile.read(filename)
     array = np.copy(array)
     array = speedx(array,0.99)
     wavfile.write("temp.wav", fps, (1*array).astype('int16'))
     array = pydub.audio_segment.AudioSegment.from_wav("temp.wav")
     newarray = pydub.effects.pan(array, 0.4)
     newarray.export("temp.wav", format="wav")
     filen = "temp.wav"
     wav_data = load_audiofile(filen) #should be filen
     preds = sess.run(output_node, feed_dict = {
                    'wav_data:0': wav_data
                    })
     pan = preds[0]

     array = pydub.audio_segment.AudioSegment.from_wav(filename)
     newarray = pydub.effects.high_pass_filter(array, 40)
     newarray = pydub.effects.low_pass_filter(newarray, 8000)
     newarray.export("temp.wav", format="wav")
     wav_data = load_audiofile("temp.wav")
     preds = sess.run(output_node, feed_dict = {
                     'wav_data:0': wav_data
                     })
     lpf = preds[0]
     return((nothing, pan, lpf))

def fast_ensemble6(index, sess, output_node, advben, tdir):
     nothing = get_logits(index, sess, output_node, advben, "raw", tdir)
     pan = get_logits(index, sess, output_node, advben, "pan", tdir)
     lpf = get_logits(index, sess, output_node, advben, "bpf", tdir)
     return((nothing, pan, lpf))


def detection_test(sess, output_node, score, ensemble, clf, clf2):
    accurate_count = 0
    master_count = 0
    for src in range(2,12):
        for trgt in range(2,12): #change back to 12 when complete.
            #print("\rEvaluation Progress: %d" %((src-2)*10 + (trgt-2)) + "%", end=" ")
            case_dir = format("../output/result/%s/%s" %(key[trgt], key[src]))
            if os.path.exists(case_dir):
                wav_files =[format('%s/%s' %(case_dir, f)) for f in os.listdir(case_dir) if f.endswith('.wav')]
                count = 0
                for wav_filename in wav_files:
                    if(count <= 9):
                        count+=1; continue
                    logits = ensemble(wav_filename, sess, output_node)
                    l2score = score(logits)
                    if numtrees == 2: # aggressive
                        if(clf.predict([l2score]) == [[0]] or clf2.predict([second_l1(logits)]) == [[0]]):
                            accurate_count += 1
                    elif numtrees == 3: # passive
                        if(clf.predict([l2score]) == [[0]] and clf2.predict([second_l1(logits)]) == [[0]]):
                            accurate_count += 1
                    else: 
                        if clf.predict([l2score]) == [[0]]:
                            accurate_count+=1
                    count+=1; master_count+=1
    adversarial_accuracy = accurate_count/master_count
    accurate_ben = 0; ben_count = 0
    for lbl in range(2,12):
        count = 0
        case_dir = format("data/%s" %(key[lbl]))
        if os.path.exists(case_dir):
            wav_files =[format('%s/%s' %(case_dir, f)) for f in os.listdir(case_dir) if f.endswith('.wav')]
            for wav_filename in wav_files:
                if count < 90:
                    count+=1; continue
                if count >= 180:
                    break
                logits = ensemble(wav_filename, sess, output_node)
                l2score = score(logits) 
                if numtrees == 2: # aggressive
                    if clf.predict([l2score]) == [[1]] and clf2.predict([second_l1(logits)]) == [[1]]:
                        accurate_count+=1; accurate_ben+=1
                elif numtrees == 3: # passive # elif the one before too
                    if clf.predict([l2score]) == [[1]] or clf2.predict([second_l1(logits)]) == [[1]]:
                        accurate_count+=1; accurate_ben+=1
                else:
                    if clf.predict([l2score]) == [[1]]:
                        accurate_count+=1; accurate_ben+=1
                count+=1; master_count+=1; ben_count+=1
    benign_accuracy = accurate_ben/ben_count
    return((adversarial_accuracy, benign_accuracy, accurate_count/master_count))

def all_max_pairwise_diff(logits):
    diffs = []
    for i in range(len(logits[0])):
        diff = np.max([abs(logits[0][i] - logits[1][i]), abs(logits[0][i] - logits[2][i]), abs(logits[1][i] - logits[2][i])])
        diffs.append(diff)
    return(diffs)

def all_sum_pairwise_diff(logits):
    diffs = []
    for i in range(len(logits[0])):
        pairwise_diff = abs(logits[0][i] - logits[1][i])
        pairwise_diff+= abs(logits[0][i] - logits[2][i])
        pairwise_diff+= abs(logits[1][i] - logits[2][i])
        diffs.append(pairwise_diff)
    return(diffs)

def all_variance_scoring(logits):
    variances = []
    for i in range(len(logits[0])):
        var = np.var([logits[0][i], logits[1][i], logits[2][i]])
        variances.append(var)
    return(variances)

def all_diff_scoring(logits):
    diffs = []
    for i in range(len(logits[0])):
        diff = abs(logits[0][i] - logits[1][i])
        diffs.append(diff)
    return(diffs)

def max_diff_scoring(logits):
    return([np.max(all_diff_scoring(logits))])

def max_variance_scoring(logits):
    maxvar = 0
    for i in range(len(logits[0])):
        var = np.var([logits[0][i], logits[1][i], logits[2][i]])
        if var > maxvar:
            maxvar = var
    return([maxvar])

def single_l1(logits):
    return([np.max([L1(logits[0],logits[1])])])

def second_l1(logits):
    return([np.max([L1(logits[0],logits[2])])])

def l1_scoring(logits):
    return([np.max([L1(logits[0],logits[1]),L1(logits[0],logits[2]), L1(logits[0],logits[2])])])

def maxvar_l1_scoring(logits):
    return(l1_scoring(logits) + max_variance_scoring(logits)) 

def passive_detection(filename, sess, output_node, ensemble, clf, clf2, advben, tdir):
    logits = ensemble(filename, sess, output_node, advben, tdir)
    s1 = single_l1(logits)
    s2 = second_l1(logits)
    if(clf.predict([s1]) == [[0]] and clf2.predict([s2]) == [[0]]):
        return(0)
    return(1)

def aggressive_detection(filename, sess, output_node, ensemble, clf, clf2, advben, tdir):
    logits = ensemble(filename, sess, output_node, advben, tdir)
    s1 = single_l1(logits)
    s2 = second_l1(logits)
    if(clf.predict([s1]) == [[0]] or clf2.predict([s2]) == [[0]]):
        return(0)
    return(1)

def adv_ben_save(adv, benign, tdir):
    counter = 0
    for filename in adv:
        path = format("%s/raw/adv/%d.wav" %(tdir, counter))
        array = pydub.audio_segment.AudioSegment.from_wav(filename)
        array.export(path, format="wav")
        counter+=1
    counter = 0
    for filename in benign:
        path = format("%s/raw/benign/%d.wav" %(tdir, counter))
        array = pydub.audio_segment.AudioSegment.from_wav(filename)
        array.export(path, format="wav")
        counter+=1
   
def mp3_save(adv, benign, tdir):
    counter = 0
    for filename in adv:
        path = format("%s/mp3/adv/%d.wav" %(tdir, counter))
        subprocess.call(['ffmpeg', '-i', filename, 'temp.mp3', '-y'])
        subprocess.call(['ffmpeg', '-i', 'temp.mp3', path, '-y']) 
        counter+=1
    counter = 0
    for filename in benign:
        path = format("%s/mp3/benign/%d.wav" %(tdir, counter))
        subprocess.call(['ffmpeg', '-i', filename, 'temp.mp3', '-y'])
        subprocess.call(['ffmpeg', '-i', 'temp.mp3', path, '-y'])
        counter+=1

def aac_save(adv, benign, tdir):
    counter = 0
    for filename in adv:
        if(filename == "../output/result/off/left/67961766_nohash_1.wav"):
            path = format("%s/mp3/adv/%d.wav" %(tdir, counter))
            subprocess.call(['ffmpeg', '-i', filename, 'temp.mp3', '-y'])
            subprocess.call(['ffmpeg', '-i', 'temp.mp3', path, '-y'])
            counter+=1; continue
        path = format("%s/aac/adv/%d.wav" %(tdir, counter))
        subprocess.call(['ffmpeg', '-i', filename, '-strict', '-2', 'temp.aac', '-y'])
        subprocess.call(['ffmpeg', '-i', 'temp.aac', '-strict', '-2',  path, '-y'])
        counter+=1
    counter = 0
    for filename in benign:
        path = format("%s/aac/benign/%d.wav" %(tdir, counter))
        subprocess.call(['ffmpeg', '-i', filename, '-strict', '-2', 'temp.aac', '-y'])
        subprocess.call(['ffmpeg', '-i', 'temp.aac', '-strict', '-2',  path, '-y'])
        counter+=1

def paulstretch_save(adv, benign, tdir):
    counter = 0
    for filename in adv:
        path = format("%s/paulstretch/adv/%d.wav" %(tdir, counter))
        (samplerate, smp) = load_wav(filename)
        paulstretch(samplerate, smp, 1.01, 0.11, path)
        counter+=1
    counter = 0
    for filename in benign:
        path = format("%s/paulstretch/benign/%d.wav" %(tdir, counter))
        (samplerate, smp) = load_wav(filename)
        paulstretch(samplerate, smp, 1.01, 0.11, path)
        counter+=1

def opus_save(adv, benign, tdir):
    counter = 0
    for filename in adv:
        path = format("%s/opus/adv/%d.wav" %(tdir, counter))
        os.system(format("wine ../opuswin64/opusenc.exe  %s temp.opus --quiet --bitrate 4 --expect-loss 30 --framesize 10; wine ../opuswin64/opusdec.exe temp.opus %s --quiet" %(filename, path)))
        counter+=1
    counter = 0
    for filename in benign:
        path = format("%s/opus/benign/%d.wav" %(tdir, counter))
        os.system(format("wine ../opuswin64/opusenc.exe  %s temp.opus --quiet --bitrate 4 --expect-loss 30 --framesize 10; wine ../opuswin64/opusdec.exe temp.opus %s --quiet" %(filename, path)))
        counter+=1

def speex_save(adv, benign, tdir):
    counter = 0
    for filename in adv:
        path = format("%s/speex/adv/%d.wav" %(tdir, counter))
        os.system(format("wine ../speexwin32/bin/speexenc.exe %s temp.spx --vad --narrowband --comp 8 --quality 5; wine ../speexwin32/bin/speexdec.exe temp.spx %s" %(filename, path)))
        counter+=1
    counter=0
    for filename in benign:
        path = format("%s/speex/benign/%d.wav" %(tdir, counter))
        os.system(format("wine ../speexwin32/bin/speexenc.exe %s temp.spx --vad --narrowband --comp 8 --quality 5; wine ../speexwin32/bin/speexdec.exe temp.spx %s" %(filename, path)))
        counter+=1
def pan_save(adv, benign, tdir):
    counter = 0
    for filename in adv:
        path = format("%s/pan/adv/%d.wav" %(tdir, counter))
        fps, array = wavfile.read(filename)
        array = np.copy(array)
        array = speedx(array,0.99)
        wavfile.write("temp.wav", fps, (1*array).astype('int16'))
        array = pydub.audio_segment.AudioSegment.from_wav("temp.wav")
        newarray = pydub.effects.pan(array, 0.4)
        newarray.export(path, format="wav")
        counter+=1
    counter=0
    for filename in benign:
        path = format("%s/pan/benign/%d.wav" %(tdir, counter))
        fps, array = wavfile.read(filename)
        array = np.copy(array)
        array = speedx(array,0.99)
        wavfile.write("temp.wav", fps, (1*array).astype('int16'))
        array = pydub.audio_segment.AudioSegment.from_wav("temp.wav")
        newarray = pydub.effects.pan(array, 0.4)
        newarray.export(path, format="wav")
        counter+=1

def bpf_save(adv, benign, tdir):
    counter = 0
    for filename in adv:
        path = format("%s/bpf/adv/%d.wav" %(tdir, counter))
        array = pydub.audio_segment.AudioSegment.from_wav(filename)
        newarray = pydub.effects.high_pass_filter(array, 40)
        newarray = pydub.effects.low_pass_filter(newarray, 8000)
        newarray.export(path, format="wav")
        counter+=1
    counter=0
    for filename in benign:
        path = format("%s/bpf/benign/%d.wav" %(tdir, counter))
        array = pydub.audio_segment.AudioSegment.from_wav(filename)
        newarray = pydub.effects.high_pass_filter(array, 40)
        newarray = pydub.effects.low_pass_filter(newarray, 8000)
        newarray.export(path, format="wav")
        counter+=1

def save_preprocessed(advtrain, benigntrain, advtest, benigntest):
    #adv_ben_save(advtrain, benigntrain, "train")
    #adv_ben_save(advtest, benigntest, "test")
    #speex_save(advtrain, benigntrain, "train")
    #speex_save(advtest, benigntest, "test")
    #pan_save(advtrain, benigntrain, "train")
    #pan_save(advtest, benigntest, "test")
    #bpf_save(advtrain, benigntrain, "train")
    #bpf_save(advtest, benigntest, "test")
    #mp3_save(advtrain, benigntrain, "train")
    #mp3_save(advtest, benigntest, "test")
    aac_save(advtrain, benigntrain, "train")
    aac_save(advtest, benigntest, "test")
   # paulstretch_save(advtrain, benigntrain, "train")
   # paulstretch_save(advtest, benigntest, "test")
   # opus_save(advtrain, benigntrain, "train")
   # opus_save(advtest, benigntest, "test")

def save_test_train_data():
    advtrain = []; advtest = []; benigntrain = []; benigntest = []
    for src in range(2,12):
        for trgt in range(2,12):
            case_dir = format("../output/result/%s/%s" %(key[trgt], key[src]))
            if os.path.exists(case_dir):
                advtrain+=([format('%s/%s' %(case_dir, f)) for f in os.listdir(case_dir) if f.endswith('.wav')][0:10])
                advtest+=([format('%s/%s' %(case_dir, f)) for f in os.listdir(case_dir) if f.endswith('.wav')][10::])
    for lbl in range(2,12):
        case_dir = format("data/%s" %(key[lbl]))
        if os.path.exists(case_dir):
            benigntrain+=([format('%s/%s' %(case_dir, f)) for f in os.listdir(case_dir) if f.endswith('.wav')][90:361])
            benigntest+=([format('%s/%s' %(case_dir, f)) for f in os.listdir(case_dir) if f.endswith('.wav')][0:90])
    save_preprocessed(advtrain, benigntrain, advtest, benigntest)


def index_to_orig(index):
    src = int(index/90) + 2
    trgt = int((index % 90)/10) + 2
    if(trgt >= src):
        trgt+=1
    adv_dir = format("../output/result/%s/%s" %(key[trgt], key[src]))
    filename = ([f for f in os.listdir(adv_dir) if f.endswith('.wav')][10::])[index % 10]
    return(format("../output/data/%s/%s" %(key[src], filename)))

def index_to_atk_success(index, sess, output_node):
    src = int(index/90) + 2
    trgt = int((index % 90)/10) + 2
    wav_data = load_audiofile(format("test/raw/adv/%d.wav" %(index)))
    preds = sess.run(output_node, feed_dict = {
                      'wav_data:0': wav_data
                      })
    if(np.argmax(preds[0]) == trgt):
        return(True)
    return(False)
    
def index_to_ben_success(index, sess, output_node):
     src = int(index/90) + 2
     wav_data = load_audiofile(format("test/raw/benign/%d.wav" %(index)))
     preds = sess.run(output_node, feed_dict = {
                      'wav_data:0': wav_data
                      })
     if(np.argmax(preds[0]) == src):
         return(True)
     return(False)

if(saving):
    save_test_train_data()
    exit(0)

def passive_aggressive_train(adv, benign, sess, output_node, ensemble):
    adv_scores1 = []; ben_scores1 = []; adv_scores2 = []; ben_scores2 = []
    for filename in adv:
        logits = ensemble(filename, sess, output_node, "adv", "train")
        adv_scores1.append(single_l1(logits))
        adv_scores2.append(second_l1(logits))
    for filename in benign:
        logits = ensemble(filename, sess, output_node, "benign", "train")
        ben_scores1.append(single_l1(logits))
        ben_scores2.append(second_l1(logits))
    total1 = adv_scores1 + ben_scores1
    labels = [0 for k in adv_scores1] + [1 for k in ben_scores1]
    clf = tree.DecisionTreeClassifier(criterion = "gini", max_depth=1)
    clf = clf.fit(total1,labels)
    total2 = adv_scores2 + ben_scores2
    labels = [0 for k in adv_scores2] + [1 for k in ben_scores2]
    clf2 = tree.DecisionTreeClassifier(criterion = "gini", max_depth=1)
    clf2 = clf2.fit(total2,labels)
    return((clf,clf2))

def misc_score_train(adv, benign, sess, output_node, ensemble, score):
    adv_scores1 = []; ben_scores1 = []; adv_scores2 = []; ben_scores2 = []
    for filename in adv:
        logits = ensemble(filename, sess, output_node)
        adv_scores1.append(score(logits))
        adv_scores2.append(second_l1(logits))
    for filename in benign:
        logits = ensemble(filename, sess, output_node)
        ben_scores1.append(score(logits))
        ben_scores2.append(second_l1(logits))
    total1 = adv_scores1 + ben_scores1
    labels = [0 for k in adv_scores1] + [1 for k in ben_scores1]
    clf = tree.DecisionTreeClassifier(criterion = "gini", max_depth=1)
    clf = clf.fit(total1,labels)
    total2 = adv_scores2 + ben_scores2
    labels = [0 for k in adv_scores2] + [1 for k in ben_scores2]
    clf2 = tree.DecisionTreeClassifier(criterion = "gini", max_depth=1)
    clf2 = clf2.fit(total2,labels)
    return((clf,clf2)) 

def simple_l1_trainer(adv, benign, sess, output_node, ensemble):
    adv_scores1 = []; ben_scores1 = []
    for filename in adv:
        logits = ensemble(filename, sess, output_node, "adv", "train")
        adv_scores1.append(single_l1(logits))
    for filename in benign:
        logits = ensemble(filename, sess, output_node, "benign", "train")
        ben_scores1.append(single_l1(logits))
    total1 = adv_scores1 + ben_scores1
    labels = [0 for k in adv_scores1] + [1 for k in ben_scores1]
    clf = tree.DecisionTreeClassifier(criterion = "gini", max_depth=1)
    clf = clf.fit(total1,labels)
    return(clf)

def full_l1_scoring(logits):
     n = len(logits); maxscore = 0
     for i in range(1,n):
         #for j in range(i+1, n):
         score = L1(logits[0], logits[i])
         if(score > maxscore):
                maxscore = score
     return [maxscore]

def summed_absolute_differences(logits):
    n = len(logits); diffs = [0,0,0,0,0,0,0,0,0,0]
    for i in range(1,n):
        for j in range(10):
            diffs[j] += abs(logits[i][j] - logits[0][j])
    return(diffs)

def full(logits):
    ret = []
    for vector in logits:
        ret+=[k for k in vector]
    return(ret)

def passive_aggressive_classify4(index, sess, output_node, advben, tdir):
    nothing = get_logits(index, sess, output_node, advben, "raw", tdir)
    speex = get_logits(index, sess, output_node, advben, "speex", tdir)
    pan = get_logits(index, sess, output_node, advben, "pan", tdir)
    bpf = get_logits(index, sess, output_node, advben, "bpf", tdir)
    advotes = 0; benvotes = 0
   # if(bpftree.predict([[L1(nothing, bpf)]]) == [[1]]):
   #     return((1,0))
   # if(pantree.predict([[L1(nothing, pan)]]) == [[1]]):
   #     return((1,1))
   # if(sptree.predict([[L1(nothing, speex)]]) == [[1]]):
   #     return((1,2))    
    if(np.argmax(nothing) != np.argmax(speex)):
        advotes+=1
    else:
        benvotes+=1
    if(np.argmax(nothing) != np.argmax(pan)):
        advotes+=1
    else:
        benvotes+=1
    if(np.argmax(nothing) != np.argmax(bpf)):
        advotes+=1
    else:
        benvotes+=1
    if(advotes > 0):
        return((0,3))
    return((1,3))

def passive_aggressive_train1(sess, output_node):
    benign = []; pasbenign = []; aggbenign = []
    adv = []
    #for src in range(2,12):
    #    for trgt in range(2,12):
    #        case_dir = format("../output/result/%s/%s" %(key[trgt], key[src]))
    #        if os.path.exists(case_dir):
    #            adv+=([format('%s/%s' %(case_dir, f)) for f in os.listdir(case_dir) if f.endswith('.wav')][0:10])
    #for lbl in range(2,12):
    #    case_dir = format("data/%s" %(key[lbl]))
    #    if os.path.exists(case_dir):
    #        benign+=([format('%s/%s' %(case_dir, f)) for f in os.listdir(case_dir) if f.endswith('.wav')][90:181])
    #        pasbenign+=([format('%s/%s' %(case_dir, f)) for f in os.listdir(case_dir) if f.endswith('.wav')][90:361])
    #        aggbenign+=([format('%s/%s' %(case_dir, f)) for f in os.listdir(case_dir) if f.endswith('.wav')][90:111])
    
    adv = [k for k in range(900)]
    passivism = 180; aggressivism = 6; neutral = 90
    benign = []; pasbenign = []; aggbenign = []; i = 0
    #while(i < 2710 - passivism):
    #    pasbenign+=[k for k in range(i,i+passivism)]
    #    i+=271
    #i=0
    #while(i < 2710 - aggressivism):
    #    aggbenign+=[k for k in range(i,i+aggressivism)]
    #    i+=271
    i=0
    while(i < 2710 - neutral):
        benign+=[k for k in range(i,i+neutral)]
        i+=271
    
    #benincorrect = []; advincorrect = []
    benscores = []; advscores = []

    for index in benign:
         advben = "benign"; tdir = "train"
         logits = [get_logits(index, sess, output_node, advben, "raw", tdir), get_logits(index, sess, output_node, advben, "mp3", tdir), get_logits(index, sess, output_node, advben, "aac", tdir), get_logits(index, sess, output_node, advben, "bpf", tdir), get_logits(index, sess, output_node, advben, "pan", tdir), get_logits(index, sess, output_node, advben, "opus", tdir), get_logits(index, sess, output_node, advben, "speex", tdir)]
         benscores.append(summed_absolute_differences([softmax(k) for k in logits]))


    for index in adv:
         advben = "adv"; tdir = "train"
         logits = [get_logits(index, sess, output_node, advben, "raw", tdir), get_logits(index, sess, output_node, advben, "mp3", tdir), get_logits(index, sess, output_node, advben, "aac", tdir), get_logits(index, sess, output_node, advben, "bpf", tdir), get_logits(index, sess, output_node, advben, "pan", tdir), get_logits(index, sess, output_node, advben, "opus", tdir), get_logits(index, sess, output_node, advben, "speex", tdir)]
         advscores.append(summed_absolute_differences([softmax(k) for k in logits]))

    labels = [0 for k in adv] + [1 for k in benign]
    scores = advscores + benscores
   
    # thiner
    print("Shape:", np.asarray(scores).shape) 
    clf = RandomForestClassifier()
# DecisionTreeClassifier(max_depth=1, criterion="entropy")
    clf = clf.fit(np.asarray(scores),np.asarray(labels))
    return(clf)

    aggtree = simple_l1_trainer(adv, benincorrect, sess, output_node, fast_ensemble5)
    pantree = simple_l1_trainer(adv, benincorrect, sess, output_node, fast_ensemble6)
    speextree = simple_l1_trainer(adv, benincorrect, sess, output_node, fast_ensemble4)
    return((aggtree, pantree, speextree))

    correct = 0; benign2 = []

    for filename in benign:
        score = single_l1(fast_ensemble5(filename, sess, output_node, "benign", "train"))
        if(aggtree.predict([score]) == 1):
            correct+=1
        else:
            benign2.append(filename)
 
    random.shuffle(benign2)
    pantree = simple_l1_trainer(adv, benign2[:int(min(len(benign2), np.ceil(len(adv)/30)))], sess, output_node, fast_ensemble6)
    speextree = simple_l1_trainer(adv, benign2[:int(min(len(benign2), np.ceil(len(adv)/30)))], sess, output_node, fast_ensemble4)
    return((aggtree, pantree, speextree))

    correct = 0
    for filename in adv:
        score = single_l1(fast_ensemble4(filename, sess, output_node, "adv", "test"))
        if(aggtree.predict([score]) == 0):
            correct+=1
        
    print("Adversarial accuracy:", correct/len(adv))
    correct = 0

    for filename in range(900):
        score = single_l1(fast_ensemble4(filename, sess, output_node, "benign", "test"))
        if(aggtree.predict([score]) == 1):
            correct+=1
    print("Benign Accuracy:", correct/len(benign))
    exit(0)

    trees = passive_aggressive_train(adv, benign, sess, output_node, fast_ensemble4)
    aggtrees = passive_aggressive_train(adv, aggbenign, sess, output_node, fast_ensemble4)
    pastrees = passive_aggressive_train(adv, pasbenign, sess, output_node, fast_ensemble4)
    #pastrees = trees; aggtrees = trees

    bc = 0; ac = 0
    
    bencorrect = 0; benincorrect = 0
    advcorrect = 0; advincorrect = 0
    adv2 = []; benign2 = [];
    
    for filename in adv:
        if(passive_detection(filename, sess, output_node, fast_ensemble4, pastrees[0], pastrees[1], "adv", "train") == 0):
           advcorrect+=1; continue
        if(aggressive_detection(filename, sess, output_node, fast_ensemble4, aggtrees[0], aggtrees[1], "adv", "train") == 1):
           advincorrect+=1; continue
        adv2.append(filename) # Middle Band of Uncertainty 

    for filename in benign: #benign
        if(passive_detection(filename, sess, output_node, fast_ensemble4, pastrees[0], pastrees[1], "benign", "train") == 0):
            benincorrect+=1; continue
        if(aggressive_detection(filename, sess, output_node, fast_ensemble4, aggtrees[0], aggtrees[1], "benign", "train") == 1):
            bencorrect+=1; continue
        benign2.append(filename) # Middle Band of Uncertainty

    ac += advcorrect; bc += bencorrect
    os.system("echo \"First tier of passive aggressive training\"")
    print("First tier of passive-aggressive training:")
    print("Adversarial Correct Elimination Rate:", advcorrect/len(adv))
    print("Adversarial Incorrect Elimination Rate:", advincorrect/len(adv))
    print("Benign Correct Elimination Rate:", bencorrect/len(benign))
    print("Benign Incorrect Elimination Rate:", benincorrect/len(benign))

    shuffle(benign2); shuffle(adv2)
    #aggtrees2 = passive_aggressive_train(adv2[:min(len(benign2),len(adv2))], benign2[:int(np.ceil(min(len(benign2),len(adv2))/2))], sess, output_node, ensemble5)
    #pastrees2 = passive_aggressive_train(adv2[:int(np.ceil(min(len(adv2),len(benign2))/2))], benign2[:min(len(benign2),len(adv2))], sess, output_node, ensemble5)
    #trees2 = passive_aggressive_train(adv, benign, sess, output_node, fast_ensemble5) #second pair of trees.
    #aggtrees2 = trees2; pastrees2 = trees2
    aggtrees2 = passive_aggressive_train(adv, aggbenign, sess, output_node, fast_ensemble5)
    pastrees2 = passive_aggressive_train(adv, pasbenign, sess, output_node, fast_ensemble5)

    bencorrect = 0; benincorrect = 0
    advcorrect = 0; advincorrect = 0
    adv3 = []; benign3 = []

    for filename in adv2:
        if(passive_detection(filename, sess, output_node, fast_ensemble5, pastrees2[0], pastrees2[1], "adv", "train") == 0):
           advcorrect+=1; continue
        if(aggressive_detection(filename, sess, output_node, fast_ensemble5, aggtrees2[0], aggtrees2[1], "adv", "train") == 1):
           advincorrect+=1; continue
        adv3.append(filename) # Middle Band of Uncertainty 

    for filename in benign2:
        if(passive_detection(filename, sess, output_node, fast_ensemble5, pastrees2[0], pastrees2[1], "benign", "train") == 0):
            benincorrect+=1; continue
        if(aggressive_detection(filename, sess, output_node, fast_ensemble5, aggtrees2[0], aggtrees2[1], "benign", "train") == 1):
            bencorrect+=1; continue
        benign3.append(filename) # Middle Band of Uncertainty
    
    ac += advcorrect; bc += bencorrect
    print("Second tier of passive-aggressive training:")
    os.system("echo \"Second tier of passive aggressive training\"")
    print("Adversarial Correct Elimination Rate:", advcorrect/len(adv2))
    print("Adversarial Incorrect Elimination Rate:", advincorrect/len(adv2))
    print("Benign Correct Elimination Rate:", bencorrect/len(benign2))
    print("Benign Incorrect Elimination Rate:", benincorrect/len(benign2))

    #trees3 = passive_aggressive_train(adv, benign, sess, output_node, fast_ensemble6) # third pair of trees
    shuffle(benign3); shuffle(adv3)
    #pastrees3 = passive_aggressive_train(adv3[:int(np.ceil(min(len(benign3), len(adv3))/2))], benign3[:int(np.ceil(min(len(benign3),len(adv3))/1.0))], sess, output_node, ensemble6)
    #aggtrees3 = passive_aggressive_train(adv3[:min(len(benign3), len(adv3))], benign3[:int(np.ceil(min(len(benign3), len(adv3))/2))], sess, output_node, ensemble6)
    bencorrect = 0; benincorrect = 0
    advcorrect = 0; advincorrect = 0
    adv4 = []; benign4 = []
    pastrees3 = passive_aggressive_train(adv, pasbenign, sess, output_node, fast_ensemble6)
    aggtrees3 = passive_aggressive_train(adv, aggbenign, sess, output_node, fast_ensemble6)
    for filename in adv3:
        if(passive_detection(filename, sess, output_node, fast_ensemble6, pastrees3[0], pastrees3[1], "adv", "train") == 0):
           advcorrect+=1; continue
        if(aggressive_detection(filename, sess, output_node, fast_ensemble6, aggtrees3[0], aggtrees3[1], "adv", "train") == 1):
           advincorrect+=1; continue
        adv4.append(filename) # Middle Band of Uncertainty 

    for filename in benign3:
        if(passive_detection(filename, sess, output_node, fast_ensemble6, pastrees3[0], pastrees3[1], "benign", "train") == 0):
            benincorrect+=1; continue
        if(aggressive_detection(filename, sess, output_node, fast_ensemble6, aggtrees3[0], aggtrees3[1], "benign", "train") == 1):
            bencorrect+=1; continue
        benign4.append(filename) # Middle Band of Uncertainty

    ac += advcorrect; bc += bencorrect
    os.system("echo \"Third tier of passive aggressive training\"")
    if(len(adv3) > 0 and len(benign3) > 0):
        print("Third tier of passive-aggressive training:")
        print("Adversarial Correct Elimination Rate:", advcorrect/len(adv3))
        print("Adversarial Incorrect Elimination Rate:", advincorrect/len(adv3))
        print("Benign Correct Elimination Rate:", bencorrect/len(benign3))
        print("Benign Incorrect Elimination Rate:", benincorrect/len(benign3))
    
    print("Final Band of Uncertainty Size:", (len(adv4) + len(benign4)))
    print("Final Band of Uncertainty Rate:", (len(adv4) + len(benign4))/(len(adv) + len(benign)))
    #if(len(adv4) + len(benign4) > 0):
    #    print("Final Band of Uncertainty Accuracy:", len(adv4)/(len(adv4) + len(benign4)))

    #ac += len(adv4)
    
    #lasttrees = misc_score_train(adv, benign, sess, output_node, ensemble4, max_variance_scoring)
    #lasttrees = misc_score_train(adv4[:min(len(adv4),len(benign4))], benign4[:min(len(adv4),len(benign4))], sess, output_node, ensemble4, max_variance_scoring)
    lasttrees = trees
    advcorrect = 0; bencorrect = 0
    for filename in adv4:
        if(lasttrees[0].predict([single_l1(fast_ensemble4(filename, sess, output_node, "adv", "train"))]) == [[0]]):
            advcorrect+=1

    for filename in benign4:
        if(lasttrees[0].predict([single_l1(fast_ensemble4(filename, sess, output_node, "benign", "train"))]) == [[1]]):
            bencorrect+=1

    if(len(adv4) > 0 and  len(benign4) > 0):
        print("Final Band Adversarial Accuracy:", advcorrect/len(adv4))
        print("Final Band Benign Accuracy:", bencorrect/len(benign4))
    
    ac+=advcorrect
    bc+=bencorrect

    print("Total Adversarial Accuracy (training):", ac/len(adv))
    print("Total Benign Accuracy (training):", bc/len(benign)) 
    return((pastrees, aggtrees,  pastrees2, aggtrees2, pastrees3, aggtrees3, lasttrees))

def passive_aggressive_classify1(filename, sess, output_node, trees1, trees2, trees3, aggtrees1, aggtrees2, aggtrees3, lasttrees, advben):
    if(passive_detection(filename, sess, output_node, fast_ensemble4, trees1[0], trees1[1], advben, "test") == 0):
        return(0)
    if(aggressive_detection(filename, sess, output_node, fast_ensemble4, aggtrees1[0], aggtrees1[1], advben, "test") == 1):
        return(1)
    if(passive_detection(filename, sess, output_node, fast_ensemble5, trees2[0], trees2[1], advben, "test") == 0):
        return(0)
    if(aggressive_detection(filename, sess, output_node, fast_ensemble5, aggtrees2[0], aggtrees2[1], advben, "test") == 1):
        return(1)
    if(passive_detection(filename, sess, output_node, fast_ensemble6, trees3[0], trees3[1], advben, "test") == 0):
        return(0)
    if(aggressive_detection(filename, sess, output_node, fast_ensemble6, aggtrees3[0], aggtrees3[1], advben, "test") == 1):
        return(1)
    #return(0) # Final band of uncertainty, return adversarial
    logits = single_l1(fast_ensemble4(filename,sess, output_node, advben, "test"))
    if(lasttrees[0].predict([logits]) == [[0]]):
        return(0)
    elif(lasttrees[0].predict([logits]) == [[1]]):
        return(1)
    a = 3/0 # catch for bad predictions
    return(0)

def passive_aggressive_classify2(index, sess, output_node, advben):
    nothing = get_logits(index, sess, output_node, advben, "raw", "test")
    speex = get_logits(index, sess, output_node, advben, "speex", "test")
    pan = get_logits(index, sess, output_node, advben, "pan", "test")
    bpf = get_logits(index, sess, output_node, advben, "bpf", "test")
    if(np.argmax(nothing) != np.argmax(speex) and np.argmax(nothing) != np.argmax(pan)):
        return((0,0))
    if(np.argmax(nothing) == np.argmax(speex) and np.argmax(nothing) == np.argmax(pan)):
        return((1,0))
    if(np.argmax(nothing) != np.argmax(speex) and np.argmax(nothing) != np.argmax(bpf)):
        return((0,1))
    if(np.argmax(nothing) == np.argmax(speex) and np.argmax(nothing) == np.argmax(bpf)):
        return((1,1))
    if(np.argmax(nothing) != np.argmax(bpfx) and np.argmax(nothing) != np.argmax(pan)):
        return((0,2))
    if(np.argmax(nothing) == np.argmax(bpf) and np.argmax(nothing) == np.argmax(pan)):
        return((0,2))
    if(np.argmax(nothing) == np.argmax(speex)):
        return((1,3))
    return((0,3))

def passive_aggressive_classify5(index, sess, output_node, advben, tdir):
    nothing = get_logits(index, sess, output_node, advben, "raw", tdir)
    opus = get_logits(index, sess, output_node, advben, "opus", tdir)
    pan = get_logits(index, sess, output_node, advben, "pan", tdir)
    bpf = get_logits(index, sess, output_node, advben, "bpf", tdir)
    mp3 = get_logits(index, sess, output_node, advben, "mp3", tdir)
    speex = get_logits(index, sess, output_node, advben, "speex", tdir)
    aac = get_logits(index, sess, output_node, advben, "aac", tdir)
    advotes = 0; benvotes = 0
   # if(bpftree.predict([[L1(nothing, bpf)]]) == [[1]]):
   #     return((1,0))
   # if(pantree.predict([[L1(nothing, pan)]]) == [[1]]):
   #     return((1,1))
   # if(sptree.predict([[L1(nothing, speex)]]) == [[1]]):
   #     return((1,2))    
    if(np.argmax(nothing) != np.argmax(speex)):
        advotes+=1
    else:
        benvotes+=1
    if(np.argmax(nothing) != np.argmax(pan)):
        advotes+=1
    else:
        benvotes+=1
    if(np.argmax(nothing) != np.argmax(bpf)):
        advotes+=1
    else:
        benvotes+=1
    if(np.argmax(nothing) != np.argmax(mp3)):
        advotes+=1
    else:
        benvotes+=1
    if(np.argmax(nothing) != np.argmax(opus)):
        advotes+=1
    else:
        benvotes+=1
    if(np.argmax(nothing) != np.argmax(aac)):
        advotes+=1
    else:
        benvotes+=1
    if(advotes > 2):
        return((0,3))
    return((1,3))

def passive_aggressive_classify6(index, sess, output_node, advben, tdir):
    nothing = get_logits(index, sess, output_node, advben, "raw", tdir)
    speex = get_logits(index, sess, output_node, advben, "bpf", tdir)
    if(np.argmax(nothing) != np.argmax(speex)):
        return((0,3))
    return((1,3))

def passive_aggressive_classify3(index, sess, output_node, advben, tdir, bpftree, pantree, sptree):
    nothing = get_logits(index, sess, output_node, advben, "raw", tdir)
    speex = get_logits(index, sess, output_node, advben, "speex", tdir)
    pan = get_logits(index, sess, output_node, advben, "pan", tdir)
    bpf = get_logits(index, sess, output_node, advben, "bpf", tdir)
    advotes = 0; benvotes = 0
    if(bpftree.predict([[L1(nothing, bpf)]]) == [[1]]):
        return((1,0))
    if(pantree.predict([[L1(nothing, pan)]]) == [[1]]):
        return((1,1))
    if(sptree.predict([[L1(nothing, speex)]]) == [[1]]):
        return((1,2))    
    if(np.argmax(nothing) != np.argmax(speex)):
        advotes+=1
    else:
        benvotes+=1
    if(np.argmax(nothing) != np.argmax(pan)):
        advotes+=1
    else:
        benvotes+=1
    if(np.argmax(nothing) != np.argmax(bpf)):
        advotes+=1
    else:
        benvotes+=1
    if(advotes > 0):
        return((0,3))
    return((1,3))

def passive_aggressive_test1(sess, output_node, bpftree, pantree, sptree):#, trees1, trees2, trees3, aggtrees1, aggtrees2, aggtrees3, lasttrees):
    print("Testing Passive Aggressiveness")
    ac = 0; bc = 0; num_adv = 0; num_ben = 0;
   # for src in range(2,12):
   #     for trgt in range(2,12): #change back to 12 when complete.
   #         case_dir = format("../output/result/%s/%s" %(key[trgt], key[src]))
   #         if os.path.exists(case_dir):
   #             wav_files =[format('%s/%s' %(case_dir, f)) for f in os.listdir(case_dir) if f.endswith('.wav')]
   #             count = 0
   #             for wav_filename in wav_files:
   #                 if(count <= 9):
   #                     count+=1; continue
   #                 if(passive_aggressive_classify1(wav_filename, sess, output_node, trees1, trees2, trees3, aggtrees1, aggtrees2, aggtrees3, lasttrees) == 0):
   #                     ac+=1
   #                 count+=1; num_adv+=1
   # for lbl in range(2,12):
   #     count = 0
   #     case_dir = format("data/%s" %(key[lbl]))
   #     if os.path.exists(case_dir):
   #         wav_files =[format('%s/%s' %(case_dir, f)) for f in os.listdir(case_dir) if f.endswith('.wav')]
   #         for wav_filename in wav_files:
   #             if count >= 90:
   #                 break
   #             if(passive_aggressive_classify1(wav_filename, sess, output_node, trees1, trees2, trees3, aggtrees1, aggtrees2, aggtrees3, lasttrees) == 1):
   #                 bc+=1 
   #             count+=1; num_ben+=1
   
    adv = [k for k in range(900)]
    ben = [k for k in range(900)]
  
    advcorr = []; advincorr = []; bencorr = []; benincorr = []
 
    acorr = [0, 0, 0, 0]; ai = [0,0,0,0]
    bcorr = [0, 0, 0, 0]; bi = [0,0,0,0]
    for filename in adv:
         index = filename
         #if(passive_aggressive_classify1(filename, sess, output_node, trees1, trees2, trees3, aggtrees1, aggtrees2, aggtrees3, lasttrees, "adv") == 0):
         #classify = passive_aggressive_classify5(filename, sess, output_node, "adv", "train")
         #if(classify[0] == 0):
         #    ac+=1; acorr[classify[1]]+=1
         #else:
         #    ai[classify[1]]+=1
         
         classify = passive_aggressive_classify5(filename, sess, output_node, "adv", "test")
         if(classify[0] == 0):
             ac+=1; acorr[classify[1]]+=1; advcorr.append(filename)
         else:
             ai[classify[1]]+=1; advincorr.append(filename)
         #advben = "adv"; tdir = "test"
         #logits = [get_logits(index, sess, output_node, advben, "raw", tdir), get_logits(index, sess, output_node, advben, "mp3", tdir), get_logits(index, sess, output_node, advben, "aac", tdir), get_logits(index, sess, output_node, advben, "bpf", tdir), get_logits(index, sess, output_node, advben, "pan", tdir), get_logits(index, sess, output_node, advben, "opus", tdir), get_logits(index, sess, output_node, advben, "speex", tdir)]
         #if(bpftree.predict([summed_absolute_differences([softmax(k) for k in logits])]) == [[0]]):
         #    ac+=1; acorr[3]+=1
         #else:
         #    ai[3]+=1
    for filename in ben:
         index = filename
         #if(passive_aggressive_classify1(filename, sess, output_node, trees1, trees2, trees3, aggtrees1, aggtrees2, aggtrees3, lasttrees, "benign") == 1):
         #classify = passive_aggressive_classify5(filename, sess, output_node, "benign", "train")
         #if(classify[0] == 1):
         #    bc+=1; bcorr[classify[1]]+=1
         #else:
         #    bi[classify[1]]+=1
         
         classify = passive_aggressive_classify5(filename, sess, output_node, "benign", "test")
         if(classify[0] == 1):
             bc+=1; bcorr[classify[1]]+=1; bencorr.append(filename)
         else:
             bi[classify[1]]+=1; benincorr.append(filename)
         #advben = "benign"; tdir = "test"
        # logits = [get_logits(index, sess, output_node, advben, "raw", tdir), get_logits(index, sess, output_node, advben, "mp3", tdir), get_logits(index, sess, output_node, advben, "aac", tdir), get_logits(index, sess, output_node, advben, "bpf", tdir), get_logits(index, sess, output_node, advben, "pan", tdir), get_logits(index, sess, output_node, advben, "opus", tdir), get_logits(index, sess, output_node, advben, "speex", tdir)]
         #if(bpftree.predict([summed_absolute_differences([softmax(k) for k in logits])]) == [[1]]):
         #    bc+=1; bcorr[3]+=1
         #else:
         #    bi[3]+=1
    #atk_success = 0
   # adv_detects = [[0,0,0,0,0,0,0,0,0,0],
   #                [0,0,0,0,0,0,0,0,0,0],
   #                [0,0,0,0,0,0,0,0,0,0],
   #                [0,0,0,0,0,0,0,0,0,0],
   #                [0,0,0,0,0,0,0,0,0,0],
   #                [0,0,0,0,0,0,0,0,0,0],
   #                [0,0,0,0,0,0,0,0,0,0],
   #                [0,0,0,0,0,0,0,0,0,0],
   #                [0,0,0,0,0,0,0,0,0,0],
   #                [0,0,0,0,0,0,0,0,0,0]]
   # adv_total = [[0,0,0,0,0,0,0,0,0,0],
   #                [0,0,0,0,0,0,0,0,0,0],
   #                [0,0,0,0,0,0,0,0,0,0],
   #                [0,0,0,0,0,0,0,0,0,0],
   #                [0,0,0,0,0,0,0,0,0,0],
   #                [0,0,0,0,0,0,0,0,0,0],
   #                [0,0,0,0,0,0,0,0,0,0],
   #                [0,0,0,0,0,0,0,0,0,0],
   #                [0,0,0,0,0,0,0,0,0,0],
   #                [0,0,0,0,0,0,0,0,0,0]]
   # for filename in advincorr:
        #path = format("test/raw/adv/%d.wav" %(filename))
        #fps, sound = wavfile.read(path)
        #dest = format("advincorr/%d.wav" %(filename))
        #wavfile.write(dest, fps, sound)
        #path = index_to_orig(filename)
        #fps, sound = wavfile.read(path)
        #dest = format("advincorr_orig/%d.wav" %(filename))
        #wavfile.write(dest, fps, sound)
        #if(index_to_atk_success(filename, sess, output_node)):
        #    atk_success+=1
    #    index = filename
    #    src = int(index/90)
    #    trgt = int((index % 90)/10)
    #    if(trgt >= src):
    #        trgt+=1
    #    adv_total[src][trgt] += 1
    #print("Adv Incorr Targeted Success Rate:", atk_success/len(advincorr))
    #atk_success = 0
    #for filename in advcorr:
        #path = format("test/raw/adv/%d.wav" %(filename))
        #fps, sound = wavfile.read(path)
        #dest = format("advcorr/%d.wav" %(filename))
        #wavfile.write(dest, fps, sound)
        #path = index_to_orig(filename)
        #fps, sound = wavfile.read(path)
        #dest = format("advcorr_orig/%d.wav" %(filename))
        #wavfile.write(dest, fps, sound)
       # if(index_to_atk_success(filename, sess, output_node)):
       #     atk_success+=1
    #    index = filename
    #    src = int(index/90)
    #    trgt = int((index % 90)/10)
    #    if(trgt >= src):
    #        trgt+=1
    #    adv_total[src][trgt] += 1
    #    adv_detects[src][trgt] += 1
    #print("Adv Corr Targeted Sucess Rate:", atk_success/len(advcorr))
    #ben_success=0
    #for filename in bencorr:
        #path = format("test/raw/benign/%d.wav" %(filename))
        #fps, sound = wavfile.read(path)
        #dest = format("bencorr/%d.wav" %(filename))
        #wavfile.write(dest, fps, sound)
       # if(index_to_ben_success(filename, sess, output_node)):
       #     ben_success+=1
    #print("Ben Corr Classification Accuracy:", ben_success/len(bencorr))
    #ben_success=0
    #for filename in benincorr:
        #path = format("test/raw/benign/%d.wav" %(filename))
        #fps, sound = wavfile.read(path)
        #dest = format("benincorr/%d.wav" %(filename))
        #wavfile.write(dest, fps, sound)
       # if(index_to_ben_success(filename,sess,output_node)):
       #     ben_success+=1
    #print("Ben Incorr Classification Accuracy:", ben_success/len(benincorr))
    #print(adv_total)
    #print("Sanity Check:", np.sum(adv_detects)/np.sum(adv_total))
   # for i in range(len(adv_total)):
   #     for j in range(len(adv_total)):
   #         if(i==j):
   #             continue
   #         adv_detects[i][j] /= adv_total[i][j]
   # for i in adv_detects:
   #     print(i)
   # print("Sanity Check:", np.sum(adv_detects)/90)
    for i in range(4):
         print("Tier %d:" %(i))
         print("Adversarial Correct Eliminations:", acorr[i])
         print("Adversarial Incorrect Eliminations:", ai[i])
         print("Benign Correct Eliminations:", bcorr[i])
         print("Benign Incorret Eliminations:", bi[i])
    print("Adversarial Test Accuracy:", ac/900)
    print("Benign Test Accuracy:", bc/900)
    print("Total Test Accuracy:", (ac + bc)/1800)
    print("Precision:", ac/(ac + bi[3]))
    print("Recall:", ac/900)
    precision = ac/(ac + bi[3]); recall = ac/900
    f1 = 2*precision*recall/(precision + recall)
    print("F1 Score:", f1)

def passive_aggressive_algorithm():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    load_graph('ckpts/conv_actions_frozen.pb')
    with tf.Session(config=config) as sess:
        #sys.stdout = open('passive_aggressive_output.txt', 'w')
        output_node = sess.graph.get_tensor_by_name('labels_softmax:0')
        #trees = passive_aggressive_train1(sess, output_node)
        passive_aggressive_test1(sess, output_node, 0, 0, 0)#, trees[0], trees[2], trees[4], trees[1], trees[3], trees[5], trees[6])

if(passive_aggressive == 1):
    passive_aggressive_algorithm()
    exit(0)

def random_forest_train(sess, output_node):
    benign = []
    adv = []
    for src in range(2,12):
        for trgt in range(2,12):
            case_dir = format("../output/result/%s/%s" %(key[trgt], key[src]))
            if os.path.exists(case_dir):
                adv+=([format('%s/%s' %(case_dir, f)) for f in os.listdir(case_dir) if f.endswith('.wav')][0:10])
    for lbl in range(2,12):
        case_dir = format("data/%s" %(key[lbl]))
        if os.path.exists(case_dir):
            benign+=([format('%s/%s' %(case_dir, f)) for f in os.listdir(case_dir) if f.endswith('.wav')][90:181])
    adv_scores = []; ben_scores = []; 
    for filename in adv:
        logits = ensemble4(filename, sess, output_node)
        adv_scores.append(single_l1(logits) + second_l1(logits))
    for filename in benign:
        logits = ensemble4(filename, sess, output_node)
        ben_scores.append(single_l1(logits) + second_l1(logits))
    total = adv_scores + ben_scores
    labels = [0 for k in adv_scores] + [1 for k in ben_scores]
    clf = AdaBoostClassifier()
    clf.fit(total, labels)
    counter = 0 
    for i in clf.predict(total) == labels:
            if(i):
                counter+=1
    print("Training Accuracy:", counter/len(labels))
    return(clf)

def random_forest_test(sess, output_node, clf):
    ac = 0; bc = 0; num_adv = 0; num_ben = 0;
    for src in range(2,12):
        for trgt in range(2,12): #change back to 12 when complete.
            case_dir = format("../output/result/%s/%s" %(key[trgt], key[src]))
            if os.path.exists(case_dir):
                wav_files =[format('%s/%s' %(case_dir, f)) for f in os.listdir(case_dir) if f.endswith('.wav')]
                count = 0
                for filename in wav_files:
                    if(count <= 9):
                        count+=1; continue
                    logits = ensemble4(filename, sess, output_node)
                    if(clf.predict([single_l1(logits) + second_l1(logits)]) == [[0]]):
                        ac+=1
                    elif(clf.predict([single_l1(logits) + second_l1(logits)]) == [[1]]):
                        wuare = 3+3
                    else:
                        wuare = 3/0 # catch for bad prediction.
                    count+=1; num_adv+=1
    for lbl in range(2,12):
        count = 0
        case_dir = format("data/%s" %(key[lbl]))
        if os.path.exists(case_dir):
            wav_files =[format('%s/%s' %(case_dir, f)) for f in os.listdir(case_dir) if f.endswith('.wav')]
            for filename in wav_files:
                if count >= 90:
                    break
                logits = ensemble4(filename, sess, output_node)
                if(clf.predict([logits]) == [[1]]):
                    bc+=1
                count+=1; num_ben+=1

    print("Adversarial Testing Accuracy:", ac/num_adv)
    print("Benign Testing Accuracy:", bc/num_ben)
    print("Overall Testing Accuracy:", (ac + bc)/(num_adv + num_ben))

def random_forest():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    load_graph('ckpts/conv_actions_frozen.pb')
    with tf.Session(config=config) as sess:
        sys.stdout = open('random_forest_output.txt', 'w')
        output_node = sess.graph.get_tensor_by_name('labels_softmax:0')
        clf = random_forest_train(sess, output_node)
        random_forest_test(sess, output_node, clf)

if(rf == 1):
    random_forest()
    exit(0)

def adaboost_train(sess, output_node):
    benign = []
    adv = []
    for src in range(2,12):
        for trgt in range(2,12):
            case_dir = format("../output/result/%s/%s" %(key[trgt], key[src]))
            if os.path.exists(case_dir):
                adv+=([format('%s/%s' %(case_dir, f)) for f in os.listdir(case_dir) if f.endswith('.wav')][0:10])
    for lbl in range(2,12):
        case_dir = format("data/%s" %(key[lbl]))
        if os.path.exists(case_dir):
            benign+=([format('%s/%s' %(case_dir, f)) for f in os.listdir(case_dir) if f.endswith('.wav')][90:181])
    adv_scores = []; ben_scores = [];
    for filename in adv:
        logits = ensemble4(filename, sess, output_node)
        adv_scores.append(all_diff_scoring(logits))
    for filename in benign:
        logits = ensemble4(filename, sess, output_node)
        ben_scores.append(all_diff_scoring(logits))
    total = adv_scores + ben_scores
    labels = [0 for k in adv_scores] + [1 for k in ben_scores]
    clf = AdaBoostClassifier(random_state=0)
    clf.fit(total, labels)
    counter = 0
    for i in clf.predict(total) == labels:
            if(i):
                counter+=1
    print("Training Accuracy:", counter/len(labels))
    return(clf)

def adaboost_test(sess, output_node, clf):
    ac = 0; bc = 0; num_adv = 0; num_ben = 0;
    for src in range(2,12):
        for trgt in range(2,12): #change back to 12 when complete.
            case_dir = format("../output/result/%s/%s" %(key[trgt], key[src]))
            if os.path.exists(case_dir):
                wav_files =[format('%s/%s' %(case_dir, f)) for f in os.listdir(case_dir) if f.endswith('.wav')]
                count = 0
                for wav_filename in wav_files:
                    if(count <= 9):
                        count+=1; continue
                    logits = ensemble4(filename, sess, output_node)
                    if(clf.predict([all_diff_scoring(logits)]) == [[0]]):
                        ac+=1
                    elif(clf.predict([all_diff_scoring(logits)]) == [[1]]):
                        wuare = 3+3
                    else:
                        wuare = 3/0 # catch for bad prediction.
                    count+=1; num_adv+=1
    for lbl in range(2,12):
        count = 0
        case_dir = format("data/%s" %(key[lbl]))
        if os.path.exists(case_dir):
            wav_files =[format('%s/%s' %(case_dir, f)) for f in os.listdir(case_dir) if f.endswith('.wav')]
            for wav_filename in wav_files:
                if count >= 90:
                    break
                logits = ensemble4(filename, sess, output_node)
                if(clf.predict([logits]) == [[1]]):
                    bc+=1
                count+=1; num_ben+=1

    print("Adversarial Testing Accuracy:", ac/num_adv)
    print("Benign Testing Accuracy:", bc/num_ben)
    print("Overall Testing Accuracy:", (ac + bc)/(num_adv + num_ben))

def adaboost():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    load_graph('ckpts/conv_actions_frozen.pb')
    with tf.Session(config=config) as sess:
        sys.stdout = open('random_forest_output.txt', 'w')
        output_node = sess.graph.get_tensor_by_name('labels_softmax:0')
        clf = adaboost_train(sess, output_node)
        adaboost_test(sess, output_node, clf)

def detection_train(sess, output_node, score, ensemble):
    adversarial = []; adversarial2 = []
    benign = []; benign2 = []
    for src in range(2,12):
        for trgt in range(2,12): #change back to 12 when complete.
            #print("\rEvaluation Progress: %d" %((src-2)*10 + (trgt-2)) + "%", end=" ")
            case_dir = format("../output/result/%s/%s" %(key[trgt], key[src]))
            if os.path.exists(case_dir):
                wav_files =[format('%s/%s' %(case_dir, f)) for f in os.listdir(case_dir) if f.endswith('.wav')]
                count = 0
                for wav_filename in wav_files:
                    if(count > 9):
                        break
                    logits = ensemble(wav_filename, sess, output_node)
                    l2score = score(logits) 
                    adversarial.append(l2score)
                    if(numtrees >= 2):
                        adversarial2.append(second_l1(logits))
                    count+=1
    for lbl in range(2,12):
        count = 0
        case_dir = format("data/%s" %(key[lbl]))
        if os.path.exists(case_dir):
            wav_files =[format('%s/%s' %(case_dir, f)) for f in os.listdir(case_dir) if f.endswith('.wav')]
            for wav_filename in wav_files:
                if count >= 90:
                    break
                logits = ensemble(wav_filename, sess, output_node)
                l2score = score(logits)
                benign.append(l2score)
                if(numtrees >= 2):
                    benign2.append(second_l1(logits))
                count+=1
    return(adversarial, benign, adversarial2, benign2)

def predict_audio(filename, sess, output_node):
    if(ensemble == 1):
        fps, array = wavfile.read(filename)
        array = np.copy(array)
        array = speedx(array,0.99)
        wavfile.write("temp.wav", fps, (1*array).astype('int16'))
        array = pydub.audio_segment.AudioSegment.from_wav("temp.wav")
        newarray = pydub.effects.high_pass_filter(array, 40)
        newarray = pydub.effects.low_pass_filter(newarray, 8000)
        newarray.export("temp.wav", format="wav")
        filen = "temp.wav"
        wav_data = load_audiofile(filen)
        preds = sess.run(output_node, feed_dict = {
                    'wav_data:0': wav_data
                    })
        lpf = preds[0]
        
        fileo = filename.replace("output","opus_out")
        wav_data = load_audiofile(fileo)
        preds = sess.run(output_node, feed_dict = {
                    'wav_data:0': wav_data
                    })
        opus = preds[0]

        os.system(format("wine ../speexwin32/bin/speexenc.exe %s temp.spx --vad --narrowband --comp 8 --quality 5; wine ../speexwin32/bin/speexdec.exe temp.spx temp.wav" %(filename)))
        wav_data = load_audiofile(filen)
        preds = sess.run(output_node, feed_dict = {
                    'wav_data:0': wav_data
                    })
        speex = preds[0]

        probs = np.asarray(lpf) + np.asarray(opus) + np.asarray(speex)
        return(np.argmax(probs))
    if(ensemble == 2):
        os.system(format("wine ../speexwin32/bin/speexenc.exe %s temp.spx --vad --narrowband --comp 8 --quality 5; wine ../speexwin32/bin/speexdec.exe temp.spx speex.wav" %(filename)))
        filen = "temp.wav"
 
        wav_data = load_audiofile("speex.wav")
        preds = sess.run(output_node, feed_dict = {
                    'wav_data:0': wav_data
                    })
        nothing = preds[0]

        array = pydub.audio_segment.AudioSegment.from_wav("speex.wav")
        newarray = pydub.effects.high_pass_filter(array, 40)
        newarray = pydub.effects.low_pass_filter(newarray, 8000)
        newarray.export("temp.wav", format="wav")
        wav_data = load_audiofile(filen)
        preds = sess.run(output_node, feed_dict = {
                    'wav_data:0': wav_data
                    })
        lpf = preds[0]

        fps, array = wavfile.read("speex.wav")   
        array = np.copy(array)
        array = speedx(array,0.99)
        wavfile.write("temp.wav", fps, (1*array).astype('int16'))
        array = pydub.audio_segment.AudioSegment.from_wav("temp.wav")
        newarray = pydub.effects.pan(array, 0.4)
        newarray.export("temp.wav", format="wav")
        filen = "temp.wav" 
        wav_data = load_audiofile(filen)
        preds = sess.run(output_node, feed_dict = {
                    'wav_data:0': wav_data
                    })
        pan = preds[0]

        probs = np.asarray(softmax(nothing)) + np.asarray(softmax(pan)) + np.asarray(softmax(lpf))
        return(np.argmax(probs))

    filen = filename
    if(preprocessing == 1):
	# Apply stretch
        fps, array = wavfile.read(filename)
        lenfactor = 1.01
        array = stretch(array,lenfactor,2**11,2**10)
        array = np.delete(array,[0,1,-1])
        wavfile.write("temp.wav",fps, array)
        filen = "temp.wav" 
    if(preprocessing == 2): # PaulStretch
        os.system(format("./paulstretch_python/paulstretch_stereo.py --stretch=\"1.1\" --window \"0.03\"  %s temp.wav" %(filename)));
        filen = "temp.wav"
    if(preprocessing == 3):
        # Apply stretch
        fps, array = wavfile.read(filename)
        wavfile.write("temp.wav",fps, pitchshift(array, 3))
        filen = "temp.wav"
    if(preprocessing == 4):
        # Set every other element to 0.
        fps, array = wavfile.read(filename)
        array = np.copy(array)
        array[np.arange(0,len(array),2)] = 0
        wavfile.write("temp.wav",fps, array)
        filen = "temp.wav"
    if(preprocessing == 5):
        (samplerate, smp) = load_wav(filename)
        paulstretch(samplerate, smp, 1.0, 0.11, "temp.wav")
        filen = "temp.wav"
    if(preprocessing == 6):
        fps, array = wavfile.read(filename)
        array = np.copy(array)
        array = speedx(array,0.90 + np.random.random()*0.1)
        array[np.arange(0,len(array),2)] = 0
        #array = speedx(array,1/0.99)
        wavfile.write("temp.wav", fps, array)
        filen = "temp.wav"
    if(preprocessing == 7):
        return(predrandom(filename, 1/6, 5, sess))
    if(preprocessing == 8):
        fps, array = wavfile.read(filename)
        array = np.copy(array)
        array = speedx(array,0.99)
        wavfile.write("temp.wav", fps, (1*array).astype('int16'))
        filen = "temp.wav"
    if(preprocessing == 9):
        fps, array = wavfile.read(filename)
        array = np.copy(array)
        array[np.arange(0,len(array),2)] = 0
        array = stretch(array,1.01,2**11,2**10)
        wavfile.write("temp.wav", fps, array)
        filen = "temp.wav"
    if(preprocessing == 10):
        os.system(format("wine ../opuswin64/opusenc.exe  %s temp.opus --quiet --bitrate 4 --expect-loss 30 --framesize 10; wine ../opuswin64/opusdec.exe temp.opus temp.wav --quiet" %(filename)))
        filen = "temp.wav"
    if(preprocessing == 11):
        #lossrate2 = 55
        lossrate = 45 #lossrate =55 without opus compression is pretty good.
        fps, array = wavfile.read(filename)
        array = np.copy(array)
        array = speedx(array,0.99)
        #setting lossrate% of the elements to 0
        array[np.arange(0,len(array),1/(lossrate/100)).astype('int16')] = 0
        wavfile.write("temp.wav", fps, array)
        os.system(format("wine ../opuswin64/opusenc.exe  temp.wav temp.opus --quiet --bitrate 4 --expect-loss %d --framesize 2.5; wine ../opuswin64/opusdec.exe temp.opus temp.wav --quiet") %(lossrate))

        filen = "temp.wav"
    if(preprocessing == 12):
        lossrate = 45 
        fps, array = wavfile.read(filename)
        array = np.copy(array)
       # array = speedx(array,0.99)
        #setting lossrate% of the elements to 0
       # array[np.arange(0,len(array),1/(lossrate/100)).astype('int16')] = 0
        array = random_drop(array, lossrate/100)
        wavfile.write("temp.wav", fps, array)
        os.system(format("wine ../opuswin64/opusenc.exe  temp.wav temp.opus --quiet --bitrate 4 --expect-loss %d --framesize 2.5; wine ../opuswin64/opusdec.exe temp.opus temp.wav --quiet") %(lossrate))

        filen = "temp.wav"
    if(preprocessing == 13):
        os.system(format("wine ../speexwin32/bin/speexenc.exe %s temp.spx --vad --narrowband --comp 8 --quality 5; wine ../speexwin32/bin/speexdec.exe temp.spx temp.wav" %(filename)))
        filen = "temp.wav"
    if(preprocessing == 14):
        lossrate = 45 #lossrate =55 without opus compression is pretty good.
        fps, array = wavfile.read(filename)
        array = np.copy(array)
        sratio = 0.9 + np.random.random()*0.1 # Used to be 0.99
        array = speedx(array,sratio)
        #setting lossrate% of the elements to 0
        array[np.arange(0,len(array),1/(lossrate/100)).astype('int16')] = 0
        wavfile.write("temp.wav", fps, array)
        os.system(format("wine ../speexwin32/bin/speexenc.exe %s temp.spx --vad; wine ../speexwin32/bin/speexdec.exe temp.spx temp.wav" %("temp.wav")))
        filen = "temp.wav"
    if(preprocessing == 15): # MP3 Compression
        subprocess.call(['ffmpeg', '-i', filename, 'temp.mp3', '-y'])
        subprocess.call(['ffmpeg', '-i', 'temp.mp3', 'temp.wav', '-y'])
        #os.system(format("lame %s temp.mp3; mpg123 -w 16 temp.wav temp.mp3" %(filename)))
        filen = "temp.wav"
    if(preprocessing == 16): # AAC Compression
        subprocess.call(['ffmpeg', '-i', filename, '-strict', '-2', 'temp.aac', '-y'])
        subprocess.call(['ffmpeg', '-i', 'temp.aac', '-strict', '-2',  'temp.wav', '-y'])
        #os.system(format("lame %s temp.mp3; mpg123 -w 16 temp.wav temp.mp3" %(filename)))
        filen = "temp.wav"
    if(preprocessing == 17): # Low Pass Filter
        os.system(format("wine ../speexwin32/bin/speexenc.exe %s temp.spx --vad --narrowband --comp 8 --quality 5; wine ../speexwin32/bin/speexdec.exe temp.spx temp.wav" %(filename)))
        filen = "temp.wav"

        #fps, array = wavfile.read(filen)
        #array = np.copy(array)
        #array = speedx(array,0.99)
        #wavfile.write("temp.wav", fps, (1*array).astype('int16'))
        array = pydub.audio_segment.AudioSegment.from_wav("temp.wav")
        #newarray = pydub.effects.pan(array, 0.4)
        newarray = pydub.effects.high_pass_filter(array, 40)
        newarray = pydub.effects.low_pass_filter(newarray, 8000)
        newarray.export("temp.wav", format="wav")
        #fps, array = wavfile.read("temp.wav")
        #array = np.copy(array)
        #array = speedx(array,0.99)
        #wavfile.write("temp.wav", fps, (1*array).astype('int16'))
        filen = "temp.wav"
    if(preprocessing == 18):
        #os.system(format("wine ../speexwin32/bin/speexenc.exe %s temp.spx --vad --narrowband --comp 8 --quality 5; wine ../speexwin32/bin/speexdec.exe temp.spx temp.wav" %(filename)))
        array = pydub.audio_segment.AudioSegment.from_wav("temp.wav")
        newarray = pydub.effects.pan(array, 0.4)
        newarray.export("temp.wav", format="wav")
        filen = "temp.wav"
        #(samplerate, smp) = load_wav("temp.wav")
        #paulstretch(samplerate, smp, 1.0, 0.11, "temp.wav")
        fps, array = wavfile.read("temp.wav")
        array = np.copy(array)
        array = speedx(array,0.99)
        wavfile.write("temp.wav", fps, (1*array).astype('int16'))
        #array = pydub.audio_segment.AudioSegment.from_wav("temp.wav")
        #newarray = pydub.effects.pan(array, 0.4)
        #newarray.export("temp.wav", format="wav")
        #filen = "temp.wav"
    if(adversarial == 2):
        p = subprocess.Popen(['deepspeech', '../audio_adversarial_examples/models/output_graph.pb', filen, '../audio_adversarial_examples/models/alphabet.txt'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, err = p.communicate()
        return(output.decode("utf-8").replace("\n",""))
    wav_data = load_audiofile(filen)
    preds = sess.run(output_node, feed_dict = {
                    'wav_data:0': wav_data
                    })
    wav_pred = np.argmax(preds[0])
    return(wav_pred)
    if(wav_pred in range(12)):
        print("Prediction:", key[wav_pred])
    else:
        print("Prediction:", wav_pred, "is out of bounds")

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
if(detection == 1):
    load_graph('ckpts/conv_actions_frozen.pb')
    with tf.Session(config=config) as sess:
        output_node = sess.graph.get_tensor_by_name('labels_softmax:0')
        if(numtrees >= 2):
            results = detection_train(sess, output_node, single_l1, ensemble4)
            total = results[0] + results[1]
            labels = [0 for k in results[0]] + [1 for k in results[1]]
            clf = tree.DecisionTreeClassifier(criterion = "gini", max_depth=1)
            clf = clf.fit(total,labels)
            total = results[2] + results[3]
            clf2 = tree.DecisionTreeClassifier(criterion = "gini", max_depth=1)
            clf2 = clf2.fit(total,labels)
            counter = 0
            predictions = np.asarray(clf.predict(total)) + np.asarray(clf2.predict(total))
            predictions = np.ceil((predictions.astype(np.float))/2)
            for i in predictions == labels:
                if(i):
                    counter+=1
            test_acc = detection_test(sess, output_node, single_l1, ensemble4, clf, clf2)
            print("Tree train accuracy:", counter/len(total))
            print("Testing Adversarial Accuracy:", test_acc[0])
            print("Testing Benign Accuracy:", test_acc[1])
            print("Overall Testing Accuracy:", test_acc[2])
            exit(0)
        results = detection_train(sess, output_node, all_max_pairwise_diff, ensemble4)
        total = results[0] + results[1]
        labels = [0 for k in results[0]] + [1 for k in results[1]]
        #total = [[k] for k in total]
        clf = tree.DecisionTreeClassifier(criterion = "gini", max_depth=10)
        clf = clf.fit(total,labels)
        counter = 0
        for i in clf.predict(total) == labels:
            if(i):
                counter+=1
        test_acc = detection_test(sess, output_node, all_max_pairwise_diff, ensemble4, clf, 0)
        print("Tree train accuracy:", counter/len(total))
        print("Testing Adversarial Accuracy:", test_acc[0])
        print("Testing Benign Accuracy:", test_acc[1])
        print("Overall Testing Accuracy:", test_acc[2])
        exit(0)

if(adversarial == -1): # Testing on all benign examples (big!)
    totfiles = 2359 + 2353 + 3745 + 3778 + 2372 + 2375 + 3845 + 3872 + 3723 + 2377
    bar = progressbar.ProgressBar(max_value=totfiles)
    load_graph('ckpts/conv_actions_frozen.pb')
    results = np.zeros(10); master_count = 0; master_success = 0
    with tf.Session(config=config) as sess:
        output_node = sess.graph.get_tensor_by_name('labels_softmax:0')
        totcount = 0
        for lbl in range(2,12):
            if(preprocessing == 10):
                print("");
            case_dir = format("data/%s" %(key[lbl]))
            if os.path.exists(case_dir):
                wav_files =[format('%s/%s' %(case_dir, f)) for f in os.listdir(case_dir) if f.endswith('.wav')]
                count = 0; success = 0
                for wav_filename in wav_files:
                    pr = predict_audio(wav_filename, sess, output_node)
                    if(pr == lbl):
                        success+=1
                    count+=1; totcount+=1; bar.update(totcount)
                results[lbl-2] = success/count
                master_count+=count; master_success+=success;
    print("                                        \n")
    print(results)
    print("Total Accuracy:", master_success/master_count)
    exit(0)


if(adversarial == 0):
    load_graph('ckpts/conv_actions_frozen.pb')
    results = np.zeros(10); master_count = 0; master_success = 0
    with tf.Session(config=config) as sess:
        output_node = sess.graph.get_tensor_by_name('labels_softmax:0')
        for lbl in range(2,12):
            if(preprocessing == 10):
                print("");
            case_dir = format("../output/data/%s" %(key[lbl]))
            if os.path.exists(case_dir):
                wav_files =[format('%s/%s' %(case_dir, f)) for f in os.listdir(case_dir) if f.endswith('.wav')]
                count = 0; success = 0
                for wav_filename in wav_files:
                    pr = predict_audio(wav_filename, sess, output_node)
                    if(pr == lbl):
                        success+=1
                    count+=1
                results[lbl-2] = success/count
                master_count+=count; master_success+=success;
    print("                                        \n")
    print(results)
    print("Total Accuracy:", master_success/master_count)
    exit(0)


if(adversarial == 2):
    master_count = 0; master_success = 0; master_accurate = 0;
    load_graph('ckpts/conv_actions_frozen.pb')
    results = np.zeros((10,10))
    robustness = np.zeros((10,10))
    bar = progressbar.ProgressBar(max_value=1800)
    with tf.Session(config=config) as sess:
        wrongs = []
        output_node = sess.graph.get_tensor_by_name('labels_softmax:0')
        for src in range(2,12):
            for trgt in range(2,12): #change back to 12 when complete.
                #print("\rEvaluation Progress: %d" %((src-2)*10 + (trgt-2)) + "%", end=" ")
                case_dir = format("../audio_adversarial_examples/output/%s/%s" %(key[trgt], key[src]))
                if os.path.exists(case_dir):
                    wav_files =[format('%s/%s' %(case_dir, f)) for f in os.listdir(case_dir) if f.endswith('.wav')]
                    count = 0; success = 0; accurate = 0
                    for wav_filename in wav_files:
                        bar.update(master_count + count)
                        pr = predict_audio(wav_filename, sess, output_node)
                        if(pr == key[trgt]):
                            success+=1
                        if(pr == key[src]):
                            accurate+=1
                        else:
                            wrongs.append((pr, key[src]))
                        count+=1
                    results[src-2,trgt-2] = success/count
                    robustness[src-2,trgt-2] = accurate/count
                    master_count+=count; master_success+=success; master_accurate+=accurate
    print("\r                                        \n")
    print(results)
    print("Total Attack Success:", master_success/master_count)

    print("\n")
    print(robustness)
    print("Total Model Accuracy against Adversarial Examples:", master_accurate/master_count)
    print(wrongs)
    exit(0)


master_count = 0; master_success = 0; master_accurate = 0;
load_graph('ckpts/conv_actions_frozen.pb')
results = np.zeros((10,10))
robustness = np.zeros((10,10))
bar = progressbar.ProgressBar(max_value=1800)
with tf.Session(config=config) as sess:
    output_node = sess.graph.get_tensor_by_name('labels_softmax:0')
    for src in range(2,12):
        for trgt in range(2,12): #change back to 12 when complete.
            #print("\rEvaluation Progress: %d" %((src-2)*10 + (trgt-2)) + "%", end=" ")
            case_dir = format("../output/result/%s/%s" %(key[trgt], key[src]))
            if os.path.exists(case_dir):
                wav_files =[format('%s/%s' %(case_dir, f)) for f in os.listdir(case_dir) if f.endswith('.wav')]
                count = 0; success = 0; accurate = 0
                for wav_filename in wav_files:
                    if(master_count + count == 855 and preprocessing == 16):
                        count+=1; continue
                    pr = predict_audio(wav_filename, sess, output_node)
                    if(pr == trgt):
                        success+=1
                    if(pr == src):
                        accurate+=1
                    count+=1
                    bar.update(master_count + count)
                results[src-2,trgt-2] = success/count
                robustness[src-2,trgt-2] = accurate/count
                master_count+=count; master_success+=success; master_accurate+=accurate
print("\r                                        \n")
print(results)
print("Total Attack Success:", master_success/master_count)

print("\n")
print(robustness)
print("Total Model Accuracy against Adversarial Examples:", master_accurate/master_count)


