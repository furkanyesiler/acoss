# -*- coding: utf-8 -*-
"""

"""
from scipy.signal import resample
from essentia import Pool, array
import essentia.standard as estd
import numpy as np
import librosa
from sys import argv
import os


class AudioFeatures:
    """
    Class containing methods to compute various audio features
    Methods :
                chroma_stft   : Computes chromagram using short fourier transform
                chroma_cqt    : Computes chromagram from constant-q transform of the audio signal
                chroma_cens   : Computes improved chromagram using CENS method as mentioned in
                chroma_hpcp   : Computes Harmonic pitch class profiles aka HPCP (improved chromagram)
    
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
                feature = AudioFeatures("./data/test_audio.wav")
                #chroma cens with default parameters
                feature.chroma_cens()
                #chroma stft with default parameters
                feature.chroma_stft()

    TODO: add more features such as
          - MFCC
          - CENS
    """

    def __init__(self, audio_file, mono=True, hop_length=512, sample_rate=44100, normalize_gain=False):
        """"""
        self.hop_length = hop_length
        self.fs = sample_rate
        self.audio_file = audio_file
        if normalize_gain:
            self.audio_vector = estd.EasyLoader(filename=audio_file, sampleRate=self.fs, replayGain=-9)()
        elif mono:
            self.audio_vector = estd.MonoLoader(filename=audio_file, sampleRate=self.fs)()
        print ("== Audio vector of %s loaded with shape %s and sample rate %s ==" % (audio_file, self.audio_vector.shape, self.fs))

    def resample_audio(self, target_sample_rate):
        """Downsample a audio into a target sample rate"""
        if target_sample_rate > self.fs:
            raise ValueError("Target_sample_rate should be lower than %s" % self.fs)
        resampler = estd.Resample(inputSampleRate=self.fs, outputSampleRate=target_sample_rate, quality=1)
        return resampler.compute(self.audio_vector)

    def audio_slicer(self, endTime, startTime=0):
        """
        Trims the audio signal array with a specified start and end time in seconds
        :param endTime: endTime for slicing
        :param startTime: (default: 0)

        :return:
        """
        trimmer = estd.Trimmer(startTime=startTime, endTime=endTime, checkRange=True)
        return trimmer.compute(self.audio_vector)

    def librosa_noveltyfn(self):
        """
        Compute librosa's onset envelope from an input signal
        Returns
        -------
        novfn: ndarray(n_frames)
            Evaluation of the audio novelty function at each audio frame,
            in time increments equal to self.hop_length
        """
        # Include max_size=3 to make like superflux
        return librosa.onset.onset_strength(y=self.audio_vector, sr=self.fs, 
                                            hop_length=self.hop_length, max_size=3)

    def madmom_rnn_onsets(self, fps=100):
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
        {
            'tempo': float
                 Average tempo
            'onsets': ndarray(n_onsets)
                of beat intervals in number of windows
            'novfn': ndarray(n_frames)
                Evaluation of the audio novelty function at each audio frame,
                in time increments equal to self.hop_length
        }
        """
        from madmom.features.beats import RNNBeatProcessor, DBNBeatTrackingProcessor
        proc = DBNBeatTrackingProcessor(fps=fps)
        novfn = RNNBeatProcessor()(self.audio_file)
        b = proc(novfn)
        tempo = 60/np.mean(b[1::] - b[0:-1])
        onsets = np.array(np.round(b*self.fs/float(self.hop_length)), dtype=np.int64)
        # Resample the audio novelty function to correspond to the 
        # correct hop length
        nframes = len(self.librosa_noveltyfn())
        novfn = np.interp(np.arange(nframes)*self.hop_length/float(self.fs), np.arange(len(novfn))/float(fps), novfn) 
        return {'tempo':tempo, 'onsets':onsets, 'novfn':novfn}

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
        tempo, onsets = librosa.beat.beat_track(y=y_percussive,sr=self.fs,start_bpm=tempobias)
        return {'tempo':tempo, 'onsets':onsets}

    @staticmethod
    def resample_feature(feature_array, factor):
        """
        downsample a input feature array with a given step size
        """
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
        return np.swapaxes(chroma, 0, 1)

    def chroma_cqt(self, display=False):
        """
        Computes the chromagram feature from the constant-q transform of the input audio signal
        """
        chroma = librosa.feature.chroma_cqt(y=self.audio_vector,
                                            sr=self.fs,
                                            hop_length=self.hop_length)
        if display:
            display_chroma(chroma, self.hop_length)
        return np.swapaxes(chroma, 0, 1)

    def chroma_cens(self, display=False):
        '''
        Computes CENS chroma vectors for the input audio signal (numpy array)
        Refer https://librosa.github.io/librosa/generated/librosa.feature.chroma_cens.html for more parameters
        '''
        chroma_cens = librosa.feature.chroma_cens(y=self.audio_vector,
                                                  sr=self.fs,
                                                  hop_length=self.hop_length)
        if display:
            display_chroma(chroma_cens, self.hop_length)
        return np.swapaxes(chroma_cens, 0, 1)

    def hpcp(self,
            frameSize=4096,
            windowType='blackmanharris62',
            harmonicsPerPeak=8,
            magnitudeThreshold=1e-05,
            maxPeaks=1000,
            whitening=True,
            referenceFrequency=440,
            minFrequency=40,
            maxFrequency=5000,
            nonLinear=False,
            numBins=12,
            display=False):
        """
        Compute Harmonic Pitch Class Profiles (HPCP) for the input audio files using essentia standard mode using
        the default parameters as mentioned in [1].
        Please refer to the following paper for detailed explanantion of the algorithm.
        [1]. Gómez, E. (2006). Tonal Description of Polyphonic Audio for Music Content Processing.
        For full list of parameters of essentia standard mode HPCP please refer to http://essentia.upf.edu/documentation/reference/std_HPCP.html
        Parameters
            harmonicsPerPeak : (integer ∈ [0, ∞), default = 0) :
            number of harmonics for frequency contribution, 0 indicates exclusive fundamental frequency contribution
            maxFrequency : (real ∈ (0, ∞), default = 5000) :
            the maximum frequency that contributes to the HPCP [Hz] (the difference between the max and split frequencies must not be less than 200.0 Hz)

            minFrequency : (real ∈ (0, ∞), default = 40) :
            the minimum frequency that contributes to the HPCP [Hz] (the difference between the min and split frequencies must not be less than 200.0 Hz)

            nonLinear : (bool ∈ {true, false}, default = false) :
            apply non-linear post-processing to the output (use with normalized='unitMax'). Boosts values close to 1, decreases values close to 0.
            normalized (string ∈ {none, unitSum, unitMax}, default = unitMax) :
            whether to normalize the HPCP vector

            referenceFrequency : (real ∈ (0, ∞), default = 440) :
            the reference frequency for semitone index calculation, corresponding to A3 [Hz]

            sampleRate : (real ∈ (0, ∞), default = 44100) :
            the sampling rate of the audio signal [Hz]

            numBins : (integer ∈ [12, ∞), default = 12) :
            the size of the output HPCP (must be a positive nonzero multiple of 12)
            whitening : (boolean (True, False), default = False)
            Optional step of computing spectral whitening to the output from speakPeak magnitudes
        """
        audio = array(self.audio_vector)
        frameGenerator = estd.FrameGenerator(audio, frameSize=frameSize, hopSize=self.hop_length)
        window = estd.Windowing(type=windowType)
        spectrum = estd.Spectrum()
        # Refer http://essentia.upf.edu/documentation/reference/std_SpectralPeaks.html
        spectralPeaks = estd.SpectralPeaks(magnitudeThreshold=magnitudeThreshold,
                                            maxFrequency=maxFrequency,
                                            minFrequency=minFrequency,
                                            maxPeaks=maxPeaks,
                                            orderBy="frequency",
                                            sampleRate=self.fs)
        # http://essentia.upf.edu/documentation/reference/std_SpectralWhitening.html
        spectralWhitening = estd.SpectralWhitening(maxFrequency= maxFrequency,
                                                    sampleRate=self.fs)
        # http://essentia.upf.edu/documentation/reference/std_HPCP.html
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
            spectrum_mag = spectrum(window(frame))
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
            display_chroma(np.swapaxes(pool['tonal.hpcp']), 0, 1)

        return pool['tonal.hpcp']

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
        else:
            raise IOError("two_d_fft_mag: Wrong parameter 'feature type'. "
                          "Should be in one of ['audio', 'hpcp', 'chroma_cqt', 'chroma_cens']")

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


def display_chroma(chroma, hop_size=512, cmap="jet"):
    """
    Make plots for input chroma vector using 
    """
    from librosa.display import specshow
    import matplotlib.pyplot as plt
    plt.figure(figsize=(16, 8))
    plt.subplot(2,1,1)
    plt.title("Chroma")
    specshow(np.swapaxes(chroma,1,0), x_axis='time', y_axis='chroma', cmap=cmap, hop_length=hop_size)
    plt.show()
    return

if __name__ == '__main__':
    filename = argv[1]
    song = AudioFeatures(filename)
    librosa_onsets = song.librosa_onsets()['onsets']
    madmom_onsets = song.madmom_rnn_onsets()['onsets']
    song.export_onset_clicks("%s_librosa_onsets.mp3"%filename, librosa_onsets)
    song.export_onset_clicks("%s_madmom_onsets.mp3"%filename, madmom_onsets)
    