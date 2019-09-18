# -*- coding: utf-8 -*-
"""
@2019
"""
import os
import numpy as np

import essentia.standard as estd
import librosa
from essentia import Pool, array, run


class AudioFeatures(object):
    """
    Class containing methods to compute various audio features
    
    Attributes:
        hop_length: int
            Hop length between frames.  Same across all features
        fs: int
            Sample rate
        audio_file: string
            Path to audio file
        audio_vector: ndarray(N)
            List of audio samples
    
    Example use :
                >>> feature = AudioFeatures("./data/test_audio.wav")
                #chroma cens with default parameters
                >>> feature.chroma_cens()
                #chroma stft with default parameters
                >>> feature.chroma_stft()

    """

    def __init__(self, audio_file, mono=True, hop_length=512, sample_rate=44100, normalize_gain=False, verbose=False):
        """
        :param audio_file:
        :param mono:
        :param hop_length:
        :param sample_rate:
        :param normalize_gain:
        :param verbose:
        """
        self.hop_length = hop_length
        self.fs = sample_rate
        self.audio_file = audio_file
        if normalize_gain:
            self.audio_vector = estd.EasyLoader(filename=audio_file, sampleRate=self.fs, replayGain=-9)()
        elif mono:
            self.audio_vector = estd.MonoLoader(filename=audio_file, sampleRate=self.fs)()
        if verbose:
            print("== Audio vector of %s loaded with shape %s and sample rate %s =="
                  % (audio_file, self.audio_vector.shape, self.fs))

    def resample_audio(self, target_sample_rate):
        """Downsample a audio into a target sample rate"""
        if target_sample_rate > self.fs:
            raise ValueError("Target_sample_rate should be lower than %s" % self.fs)
        resampler = estd.Resample(inputSampleRate=self.fs, outputSampleRate=target_sample_rate, quality=1)
        return resampler.compute(self.audio_vector)

    def audio_slicer(self, endTime, startTime=0):
        """
        Trims the audio signal array with a specified start and end time in seconds

        Parameters
        ----------
        endTime: endTime for slicing
        startTime: (default: 0)

        Returns
        -------
        trimmed_audio: ndarray
        """
        trimmer = estd.Trimmer(startTime=startTime, endTime=endTime, checkRange=True)
        return trimmer.compute(self.audio_vector)

    def librosa_noveltyfn(self):
        """
        Compute librosa's onset envelope from an input signal.

        Returns
        -------
        novfn: ndarray(n_frames)
            Evaluation of the audio novelty function at each audio frame,
            in time increments equal to self.hop_length
        """
        # Include max_size=3 to make like superflux
        return librosa.onset.onset_strength(y=self.audio_vector, sr=self.fs, 
                                            hop_length=self.hop_length, max_size=3)

    def madmom_features(self, fps=100):
        """
        Call Madmom's implementation of RNN + DBN beat tracking. Madmom's
        results are returned in terms of seconds, but round and convert to
        be in terms of hop_size so that they line up with the features.
        The novelty function is also computed as a side effect (and is
        the bottleneck in the computation), so also return that

        Parameters
        ----------
        fps: int
            Frames per second in processing
        Returns
        -------
        output: a python dict with following key, value pairs
            {
                'tempos': ndarray(n_levels, 2)
                    An array of tempo estimates in beats per minute,
                    along with their confidences
                'onsets': ndarray(n_onsets)
                    Array of onsets, where each onset indexes into a particular window
                'novfn': ndarray(n_frames)
                    Evaluation of the rnn audio novelty function at each audio
                    frame, in time increments equal to self.hop_length
                'snovfn': ndarray(n_frames)
                    Superflux audio novelty function at each audio frame,
                    in time increments equal to self.hop_length
            }
        """
        from madmom.features.beats import RNNBeatProcessor, DBNBeatTrackingProcessor
        from madmom.features.tempo import TempoEstimationProcessor
        from madmom.features.onsets import SpectralOnsetProcessor
        from madmom.audio.filters import LogarithmicFilterbank
        beatproc = DBNBeatTrackingProcessor(fps=fps)
        tempoproc = TempoEstimationProcessor(fps=fps)
        novfn = RNNBeatProcessor()(self.audio_file) # This step is the computational bottleneck
        beats = beatproc(novfn)
        tempos = tempoproc(novfn)
        onsets = np.array(np.round(beats*self.fs/float(self.hop_length)), dtype=np.int64)
        # Resample the audio novelty function to correspond to the 
        # correct hop length
        nframes = len(self.librosa_noveltyfn())
        novfn = np.interp(np.arange(nframes)*self.hop_length/float(self.fs), np.arange(len(novfn))/float(fps), novfn) 
        
        # For good measure, also compute and return superflux
        sodf = SpectralOnsetProcessor(onset_method='superflux',
                                      fps=fps,
                                      filterbank=LogarithmicFilterbank,
                                      num_bands=24,
                                      log=np.log10)
        snovfn = sodf(self.audio_file)
        snovfn = np.interp(np.arange(nframes)*self.hop_length/float(self.fs), np.arange(len(snovfn))/float(fps), snovfn) 
        return {'tempos': tempos, 'onsets': onsets, 'novfn': novfn, 'snovfn': snovfn}

    def librosa_onsets(self, tempobias=120.0):
        """
        Call librosa's implementation of dynamic programming beat tracking
        Returns
        -------
        {
            'tempo': float
                 Average tempo
            'onsets': ndarray(n_onsets)
                of beat intervals in number of windows
        }
        """
        y_harmonic, y_percussive = librosa.effects.hpss(self.audio_vector)
        tempo, onsets = librosa.beat.beat_track(y=y_percussive, sr=self.fs, start_bpm=tempobias)
        return {'tempo': tempo, 'onsets': onsets}

    @staticmethod
    def resample_feature(feature_array, factor):
        """
        downsample a input feature array with a given step size
        """
        from scipy.signal import resample
        frames = feature_array.shape[0]
        re_size = int(np.ceil(frames / float(factor)))
        return resample(feature_array, re_size)

    def chroma_stft(self, frameSize=4096, display=False):
        """
        Computes the chromagram from the short-term fourier transform of the input audio signal
        """
        chroma = librosa.feature.chroma_stft(y=self.audio_vector,
                                            sr=self.fs,
                                            tuning=0,
                                            norm=2,
                                            hop_length=self.hop_length,
                                            n_fft=frameSize)
        if display:
            display_chroma(chroma, self.hop_length)
        return chroma.T

    def chroma_cqt(self, display=False):
        """
        Computes the chromagram feature from the constant-q transform of the input audio signal
        """
        chroma = librosa.feature.chroma_cqt(y=self.audio_vector,
                                            sr=self.fs,
                                            hop_length=self.hop_length)
        if display:
            display_chroma(chroma, self.hop_length)
        return chroma.T

    def chroma_cens(self, display=False):
        """
        Computes CENS chroma vectors for the input audio signal (numpy array)
        Refer https://librosa.github.io/librosa/generated/librosa.feature.chroma_cens.html for more parameters
        """
        chroma_cens = librosa.feature.chroma_cens(y=self.audio_vector,
                                                  sr=self.fs,
                                                  hop_length=self.hop_length)
        if display:
            display_chroma(chroma_cens, self.hop_length)
        return chroma_cens.T

    def chroma_cqt_processed(self):
        """
        Adapted from librosa docs
        https://librosa.github.io/librosa_gallery/auto_examples/plot_chroma.html
        """
        from scipy.ndimage import median_filter
        harmonic = librosa.effects.harmonic(y=self.audio_vector, margin=8)
        chroma_cqt_harm = librosa.feature.chroma_cqt(y=harmonic,
                                                     sr=self.fs,
                                                     bins_per_octave=12*3,
                                                     hop_length=self.hop_length)
        chroma_filter = np.minimum(chroma_cqt_harm,
                           librosa.decompose.nn_filter(chroma_cqt_harm,
                                                       aggregate=np.median,
                                                       metric='cosine'))
        chroma_smooth = median_filter(chroma_filter, size=(1, 9))
        return {'chroma_filtered': chroma_filter, 
                'chroma_smoothed': chroma_smooth}

    def hpcp(self,
            frameSize=4096,
            windowType='blackmanharris62',
            harmonicsPerPeak=8,
            magnitudeThreshold=0,
            maxPeaks=100,
            whitening=True,
            referenceFrequency=440,
            minFrequency=100,
            maxFrequency=3500,
            nonLinear=False,
            numBins=12,
            display=False):
        """
        Compute Harmonic Pitch Class Profiles (HPCP) for the input audio files using essentia standard mode using
        the default parameters as mentioned in [1].
        Please refer to the following paper for detailed explanantion of the algorithm.
        [1]. Gómez, E. (2006). Tonal Description of Polyphonic Audio for Music Content Processing.
        For full list of parameters of essentia standard mode HPCP 
        please refer to http://essentia.upf.edu/documentation/reference/std_HPCP.html
        
        Returns
        hpcp: ndarray(n_frames, 12)
            The HPCP coefficients at each time frame
        """
        audio = array(self.audio_vector)
        frameGenerator = estd.FrameGenerator(audio, frameSize=frameSize, hopSize=self.hop_length)
        # framecutter = estd.FrameCutter(frameSize=frameSize, hopSize=self.hop_length)
        windowing = estd.Windowing(type=windowType)
        spectrum = estd.Spectrum()
        # Refer http://essentia.upf.edu/documentation/reference/streaming_SpectralPeaks.html
        spectralPeaks = estd.SpectralPeaks(magnitudeThreshold=magnitudeThreshold,
                                            maxFrequency=maxFrequency,
                                            minFrequency=minFrequency,
                                            maxPeaks=maxPeaks,
                                            orderBy="frequency",
                                            sampleRate=self.fs)
        # http://essentia.upf.edu/documentation/reference/streaming_SpectralWhitening.html
        spectralWhitening = estd.SpectralWhitening(maxFrequency= maxFrequency,
                                                    sampleRate=self.fs)
        # http://essentia.upf.edu/documentation/reference/streaming_HPCP.html
        hpcp = estd.HPCP(sampleRate=self.fs,
                        maxFrequency=maxFrequency,
                        minFrequency=minFrequency,
                        referenceFrequency=referenceFrequency,
                        nonLinear=nonLinear,
                        harmonics=harmonicsPerPeak,
                        size=numBins)
        pool = Pool()

        #compute hpcp for each frame and add the results to the pool
        for frame in frameGenerator:
            spectrum_mag = spectrum(windowing(frame))
            frequencies, magnitudes = spectralPeaks(spectrum_mag)
            if whitening:
                w_magnitudes = spectralWhitening(spectrum_mag,
                                                frequencies,
                                                magnitudes)
                hpcp_vector = hpcp(frequencies, w_magnitudes)
            else:
                hpcp_vector = hpcp(frequencies, magnitudes)
            pool.add('tonal.hpcp',hpcp_vector)

        if display:
            display_chroma(pool['tonal.hpcp'].T, self.hop_length)

        return pool['tonal.hpcp']

    def crema(self):
        """
        Compute "convolutional and recurrent estimators for music analysis" (CREMA)
        and resample so that it's reported in hop_length intervals
        NOTE: This code is a bit finnecky, and is recommended for Python 3.5.
        Check `wrapper_cream_feature` for the actual implementation.

        Returns
        -------
        crema: ndarray(n_frames, 12)
            The crema coefficients at each frame
        """
        crema_feature = _call_func_on_python_version("3.6",
                                                     ".features",
                                                     "_wrapper_crema_feature",
                                                     [self.audio_vector, self.fs, self.hop_length])
        return crema_feature

    def two_d_fft_mag(self, feature_type='chroma_cqt', display=False):
        """
        Computes 2d - fourier transform magnitude coefficients of the input feature vector (numpy array)
        Usually fed by Constant-q transform or chroma feature vectors for cover detection tasks.
        """
        if feature_type == 'audio':
            feature_vector = self.audio_vector
        elif feature_type == 'hpcp':
            feature_vector = self.hpcp()
        elif feature_type == 'chroma_cqt':
            feature_vector = self.chroma_cqt()
        elif feature_type == 'chroma_cens':
            feature_vector = self.chroma_cens()
        elif feature_type == 'crema':
            feature_vector = self.crema()
        else:
            raise IOError("two_d_fft_mag: Wrong parameter 'feature type'. "
                          "Should be in one of these ['audio', 'hpcp', 'chroma_cqt', 'chroma_cens', 'crema']")

        # 2d fourier transform
        ndim_fft = np.fft.fft2(feature_vector)
        ndim_fft_mag = np.abs(np.fft.fftshift(ndim_fft))

        if display:
            import matplotlib.pyplot as plt
            from librosa.display import specshow
            plt.figure(figsize=(8,6))
            plt.title('2D-Fourier transform magnitude coefficients')
            specshow(ndim_fft_mag, cmap='jet')

        return ndim_fft_mag

    def key_extractor(self, 
                    frameSize=4096, 
                    hpcpSize=12, 
                    maxFrequency=3500,  
                    minFrequency=25, 
                    windowType='hann',
                    profileType='bgate',
                    pcpThreshold=0.2,
                    tuningFrequency=440,
                    weightType='cosine'):
        """
        Wrapper around essentia KeyExtractor algo. This algorithm extracts key/scale for an audio signal. 
        It computes HPCP frames for the input signal and applies key estimation using the Key algorithm.

        Refer for more details https://essentia.upf.edu/documentation/reference/streaming_KeyExtractor.html

        Returns:
                a dictionary with corresponding values for key, scale and strength

        eg: {'key': 'F', 'scale': 'major', 'strength': 0.7704258561134338}

        """
        audio = array(self.audio_vector)
        key = estd.KeyExtractor(frameSize=frameSize, hopSize=self.hop_length, tuningFrequency=tuningFrequency)
        """
        TODO: test it with new essentia update
        key = ess.KeyExtractor(frameSize=frameSize,
                               hopSize=self.hop_length,
                               sampleRate=self.fs, 
                               hpcpSize=hpcpSize,
                               maxFrequency=maxFrequency,
                               minFrequency=minFrequency,
                               windowType=windowType,
                               profileType=profileType,
                               pcpThreshold=pcpThreshold,
                               tuningFrequency=tuningFrequency,
                               weightType=weightType)
        """
        key, scale, strength = key.compute(audio)

        return {'key': key, 'scale': scale, 'strength': strength}

    def tempogram(self, win_length=384, center=True, window='hann'):
        """
        Compute the tempogram: local autocorrelation of the onset strength envelope. [1]
        [1] Grosche, Peter, Meinard Müller, and Frank Kurth. “Cyclic tempogram - A mid-level tempo
        representation for music signals.” ICASSP, 2010.

        https://librosa.github.io/librosa/generated/librosa.feature.tempogram.html
        """
        return librosa.feature.tempogram(y=self.audio_vector,
                                         sr=self.fs,
                                         onset_envelope=self.librosa_noveltyfn(),
                                         hop_length=self.hop_length,
                                         win_length=win_length,
                                         center=center,
                                         window=window)

    def cqt_nsg(self, frame_size=4096):
        """
        invertible CQT algorithm based on Non-Stationary Gabor frames
        https://mtg.github.io/essentia-labs//news/2019/02/07/invertible-constant-q/
        https://essentia.upf.edu/documentation/reference/std_NSGConstantQ.html
        """
        import essentia.pytools.spectral as epy
        cq_frames, dc_frames, nb_frames = epy.nsgcqgram(self.audio_vector, frameSize=frame_size)
        return cq_frames

    def cqt(self, fmin=None, n_bins=84, bins_per_octave=12, tuning=0.0,
              filter_scale=1, norm=1, sparsity=0.01, window='hann', scale=True, pad_mode='reflect'):
        """
        Compute the constant-Q transform implementation as in librosa
        https://librosa.github.io/librosa/generated/librosa.core.cqt.html
        """
        return librosa.core.cqt(y=self.audio_vector,
                                sr=self.fs,
                                hop_length=self.hop_length,
                                fmin=fmin,
                                n_bins=n_bins,
                                bins_per_octave=bins_per_octave,
                                tuning=tuning,
                                filter_scale=filter_scale,
                                norm=norm,
                                sparsity=sparsity,
                                window=window,
                                scale=scale,
                                pad_mode=pad_mode)

    def mfcc_htk(self, window_length=22050, nmfcc=13, n_mels=26, fmax=8000, lifterexp=22):
        """
        Get MFCCs 'the HTK way' with the help of Essentia
        https://github.com/MTG/essentia/blob/master/src/examples/tutorial/example_mfcc_the_htk_way.py
        Using all of the default parameters from there except the hop length (which shouldn't matter), and a much longer window length (which has been found to work better for covers)
        Parameters
        ----------
        window_length: int
            Length of the window to use for the STFT
        nmfcc: int
            Number of MFCC coefficients to compute
        n_mels: int
            Number of frequency bands to use
        fmax: int
            Maximum frequency
        Returns
        -------
        ndarray(nmfcc, nframes)
            An array of all of the MFCC frames
        """
        fftlen = int(2**(np.ceil(np.log(window_length)/np.log(2))))
        spectrumSize= fftlen//2+1
        zeroPadding = fftlen - window_length

        w = estd.Windowing(type = 'hamming', #  corresponds to htk default  USEHAMMING = T
                            size = window_length, 
                            zeroPadding = zeroPadding,
                            normalized = False,
                            zeroPhase = False)
        
        spectrum = estd.Spectrum(size=fftlen)
        mfcc_htk = estd.MFCC(inputSize = spectrumSize,
                            type = 'magnitude', # htk uses mel filterbank magniude
                            warpingFormula = 'htkMel', # htk's mel warping formula
                            weighting = 'linear', # computation of filter weights done in Hz domain
                            highFrequencyBound = fmax, # 8000 is htk default
                            lowFrequencyBound = 0, # corresponds to htk default
                            numberBands = n_mels, # corresponds to htk default  NUMCHANS = 26
                            numberCoefficients = nmfcc,
                            normalize = 'unit_max', # htk filter normaliation to have constant height = 1  
                            dctType = 3, # htk uses DCT type III
                            logType = 'log',
                            liftering = lifterexp) # corresponds to htk default CEPLIFTER = 22


        mfccs = []
        # startFromZero = True, validFrameThresholdRatio = 1 : the way htk computes windows
        for frame in estd.FrameGenerator(self.audio_vector, frameSize = window_length, hopSize = self.hop_length , startFromZero = True, validFrameThresholdRatio = 1):
            spect = spectrum(w(frame))
            mel_bands, mfcc_coeffs = mfcc_htk(spect)
            mfccs.append(mfcc_coeffs)
        
        return np.array(mfccs, dtype=np.float32).T
    
    def mfcc_librosa(self, window_length=22050, nmfcc=20, n_mels=40, fmax=8000, lifterexp=0.6):
        """
        Using the default parameters from C Tralie
        "Early MFCC And HPCP Fusion for Robust Cover Song Identification"
        Parameters
        ----------
        window_length: int
            Length of the window to use for the STFT
        nmfcc: int
            Number of MFCC coefficients to compute
        n_mels: int
            Number of frequency bands to use
        fmax: int
            Maximum frequency
        lifterexp: float
            Liftering exponent
        Returns
        -------
        ndarray(nmfcc, nframes)
            An array of all of the MFCC frames
        """
        S = librosa.core.stft(self.audio_vector, window_length, self.hop_length)
        M = librosa.filters.mel(self.fs, window_length, n_mels = n_mels, fmax = fmax)
        X = M.dot(np.abs(S))
        X = librosa.core.amplitude_to_db(X)
        X = np.dot(librosa.filters.dct(nmfcc, X.shape[0]), X) #Make MFCC
        #Do liftering
        coeffs = np.arange(nmfcc)**lifterexp
        coeffs[0] = 1
        X = coeffs[:, None]*X
        X = np.array(X, dtype = np.float32)
        return X

    def export_onset_clicks(self, outname, onsets):
        """
        Test a beat tracker by creating an audio file
        with little blips where the onsets are
        Parameters
        ----------
        outname: string 
            Path to the file to which to output
        onsets: ndarray(n_onsets)
            An array of onsets, in terms of the hop length
        """
        import scipy.io as sio
        import subprocess
        yaudio = np.array(self.audio_vector)
        blipsamples = int(np.round(0.02*self.fs))
        blip = np.cos(2*np.pi*np.arange(blipsamples)*440.0/self.fs)
        blip = np.array(blip*np.max(np.abs(yaudio)), dtype=yaudio.dtype)
        for idx in onsets:
            l = len(yaudio[idx*self.hop_length:idx*self.hop_length+blipsamples])
            yaudio[idx*self.hop_length:idx*self.hop_length+blipsamples] = blip[0:l]
        sio.wavfile.write("temp.wav", self.fs, yaudio)
        if os.path.exists(outname):
            os.remove(outname)
        subprocess.call(["ffmpeg", "-i", "temp.wav", outname])
        os.remove("temp.wav")

    def chromaprint(self, analysisTime=30):
        """
        This algorithm computes the fingerprint of the input signal using Chromaprint algorithm. 
        It is a wrapper of the Chromaprint library

        Returns: The chromaprints are returned as base64-encoded strings.
        """
        import essentia.streaming as ess

        vec_input = ess.VectorInput(self.audio_vector)
        chromaprinter = ess.Chromaprinter(analysisTime=analysisTime, sampleRate=self.fs)
        pool = Pool()

        vec_input.data >> chromaprinter.signal
        chromaprinter.fingerprint >> (pool, 'chromaprint')
        run(vec_input)
        return pool['chromaprint']


def display_chroma(chroma, hop_length=512, fs=44100):
    """
    Make plots for input chroma vector using librosa's spechow
    Parameters
    ----------
    chroma: ndarray(n_frames, n_chroma_bins)
        An array of chroma features
    """
    from librosa.display import specshow
    specshow(chroma.T, x_axis='time', y_axis='chroma', hop_length=hop_length, sr=fs)


def _call_func_on_python_version(Version, Module, Function, ArgumentList):
    """Wrapper to call functions across different python versions"""
    import execnet
    gw = execnet.makegateway("popen//python=python%s" % Version)
    channel = gw.remote_exec("""
        from %s import %s as the_function
        channel.send(the_function(*channel.receive()))
    """ % (Module, Function))
    channel.send(ArgumentList)
    return channel.receive()


def _wrapper_crema_feature(audio_vector, sr, hop_length):
    """wrapper callback func to run crema feature computation in different python shell"""
    import crema
    from scipy import interpolate

    model = crema.models.chord.ChordModel()
    data = model.outputs(y=audio_vector, sr=sr)
    fac = (float(sr) / 44100.0) * 4096.0 / hop_length
    times_orig = fac * np.arange(len(data['chord_bass']))
    nwins = int(np.floor(float(audio_vector.size) / hop_length))
    times_new = np.arange(nwins)
    interp = interpolate.interp1d(times_orig, data['chord_pitch'].T, kind='nearest', fill_value='extrapolate')
    return interp(times_new).T
