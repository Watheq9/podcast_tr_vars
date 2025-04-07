import os, sys
sys.path.insert(0, os.getcwd())
import time
import os
import numpy as np
import torch
import pandas as pd
import whisper
import json
import logging
import argparse
import vosk
from pydub import AudioSegment
from itertools import groupby
import soundfile as sf
from vosk import Model, KaldiRecognizer
from pydub import AudioSegment
from vosk import Model, KaldiRecognizer, SetLogLevel
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2Tokenizer
import ffmpeg
import soundfile as sf
import wave
# import whisperx



device = "cuda" if torch.cuda.is_available() else "cpu"


def initailize_logger(logger, log_file, level):
    if not len(logger.handlers):  # avoid creating more than one handler
        formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        fileHandler = logging.FileHandler(log_file)
        fileHandler.setFormatter(formatter)
        streamHandler = logging.StreamHandler()
        streamHandler.setFormatter(formatter)
        logger.setLevel(level)
        logger.addHandler(fileHandler)
        logger.addHandler(streamHandler)

    return logger


def get_logger(log_file="progress.txt", level=logging.DEBUG):
    """Returns a logger to log the info and error messages in an organized and timestampped way"""

    logger = logging.getLogger(log_file)
    logger = initailize_logger(logger, log_file, level)
    return logger


def load_model(model_name=None, model_path= None):
    # You can also init model by name or with a folder path
    # model = Model(model_name="vosk-model-en-us-0.21")
    # model = Model("models/en")

    if model_name is not None:
        model = Model(model_name=model_name)
    else:
        model = Model(model_path=model_path)

    return model

def get_file_size(file_path):
    stat_info = os.stat(file_path)
    file_size_mb = stat_info.st_size / (1024 * 1024)
    return file_size_mb


# Convert OGG to WAV using ffmpeg
def convert_ogg_to_wav(ogg_file, wav_file, method='ffmpeg'):
    
    if method == 'ffmpeg':        
        # Use ffmpeg to convert .ogg to .wav in PCM format
        ffmpeg.input(ogg_file).output(wav_file, acodec='pcm_s16le',
                                        ac=1, ar='16000').run()
    else: 
        sound = AudioSegment.from_ogg(ogg_file)
        sound = sound.set_channels(1)  # Mono channel
        sound = sound.set_frame_rate(16000)  # 16 kHz sample rate
        sound.export(wav_file, format="wav", codec="pcm_s16le")


# Convert MP3 to WAV
def convert_mp3_to_wav(mp3_file, wav_file):
    sound = AudioSegment.from_mp3(mp3_file)
    sound = sound.set_channels(1)  # Mono channel
    sound = sound.set_frame_rate(16000)  # 16 kHz sample rate
    sound.export(wav_file, format="wav", codec="pcm_s16le")



def transcribe_with_vosk(model, wav_file):
    
    # read the wave file
    wf = wave.open(wav_file, "rb")
    if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
        print("Audio file must be WAV format mono PCM.")
        return None
    
    rec = KaldiRecognizer(model, wf.getframerate())
    rec.SetWords(True)  # Enable word-level timestamps
    
    transcript = []
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            result = rec.Result()
            result_dict = json.loads(result)
            if "result" in result_dict:
                transcript.extend(result_dict['result'])
    
    # Get the final result
    final_result = rec.FinalResult()
    result_dict = json.loads(final_result)
    if "result" in result_dict:
        transcript.extend(result_dict['result'])
    
    return transcript



def transcribe_with_deepspeech(model, wav_file):
    
    # read the wave file
    wf = wave.open(wav_file, "rb")
    if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
        print("Audio file must be WAV format mono PCM.")
        return None
    


    with wave.open(wav_file, 'r') as wf:
        sample_rate = wf.getframerate()
        if sample_rate != 16000:
            raise ValueError("Audio file must be 16kHz.")
        
        # Read the entire file
        frames = wf.getnframes()
        buffer = wf.readframes(frames)
        audio_data = np.frombuffer(buffer, dtype=np.int16)
    
    # Get the metadata, which includes timing information for each word
    metadata = model.sttWithMetadata(audio_data)

    # Initialize variables to accumulate characters and their start times
    word = ''
    word_start_time = None
    transcript = []

    # Loop through each character token and process them into words
    for item in metadata.transcripts[0].tokens:
        char = item.text

        # If it's the first character of a word, save its start time
        if word == '':
            word_start_time = item.start_time

        # Add characters to form a word
        if char != ' ':  # Avoid adding spaces to the word
            word += char
        else:
            # When a space is encountered, the word is completed
            if word:
                # print(f"Word: {word}, Start Time: {word_start_time:.2f} seconds")
                transcript.append({"word": word, "time": word_start_time})
            word = ''  # Reset the word

    # In case the last word doesn't end with a space, print it
    if word:
        # print(f"Word: {word}, Start Time: {word_start_time:.2f} seconds")
        transcript.append({"word": word, "time": word_start_time})
    
    return transcript



def transribe_with_wav2vec( model, audio_wav, tokenizer):

    
    speech, sample_rate = sf.read(audio_wav)

    # Wav2Vec 2.0 works with 16kHz, so ensure the sample rate is correct
    # assert sample_rate == 16000, "Wav2Vec 2.0 requires 16 kHz audio input"

    batch_duration = 240  # Set the batch size (e.g., 10 seconds)

    # Split the audio into batches (chunks of 240 seconds)
    batch_size = int(batch_duration * sample_rate)  # Number of samples in 240 seconds
    num_batches = len(speech) // batch_size
    transcription_with_timestamp = []

    chunk_start_time = 0 
    full_transcript = ""

    for i in range(num_batches + 1):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(speech))  # Handle the last chunk which might be smaller
        audio_chunk = speech[start_idx:end_idx]

        
        # Transcribe the batch
        input_values = tokenizer(audio_chunk, return_tensors="pt", padding="longest").input_values.to(device)
        # Perform inference and get logits
        with torch.no_grad():
            logits = model(input_values).logits
        # Get the predicted ids from logits  and decode
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = tokenizer.batch_decode(predicted_ids)[0].lower()


        ##############
        # this is where the logic starts to get the start and end timestamp for each word
        # ref: https://github.com/huggingface/transformers/issues/11307
        ##############
        words = [w for w in transcription.split(' ') if len(w) > 0]
        predicted_ids = predicted_ids[0].tolist()
        duration_sec = input_values.shape[1] / sample_rate

        ids_w_time = [(i / len(predicted_ids) * duration_sec, _id) for i, _id in enumerate(predicted_ids)]
        # remove entries which are just "padding" (i.e. no characers are recognized)
        ids_w_time = [i for i in ids_w_time if i[1] != tokenizer.pad_token_id]


        # now split the ids into groups of ids where each group represents a word
        split_ids_w_time = [list(group) for k, group
                            in groupby(ids_w_time, lambda x: x[1] == tokenizer.word_delimiter_token_id)
                            if not k]

        if len(split_ids_w_time) != len(words):  # make sure that there are the same number of id-groups as words. Otherwise something is wrong
            return None

        full_transcript += " "+ transcription
        for cur_ids_w_time, cur_word in zip(split_ids_w_time, words):
            _times = [_time for _time, _id in cur_ids_w_time]
            start = chunk_start_time + min(_times)
            end = chunk_start_time + max(_times)
            transcription_with_timestamp.append({"word": cur_word, "start": start, "end": end})
        
        chunk_start_time += batch_duration 

    transcription_with_timestamp.append({"full_transcription": full_transcript})
    return transcription_with_timestamp


def transcribe_with_silero(model, audio_wav, decoder, prepare_model_input, read_batch, logger=None,
                           sample_rate = 16000, batch_duration = 30 ):

    full_speech_vec = prepare_model_input(read_batch([audio_wav]), device=device)
    speech = full_speech_vec[0]
    # Set the batch size to 30 sec (0.5 minutes)

    batch_size = int(batch_duration * sample_rate) 
    num_batches = len(speech) // batch_size
    transcript_with_timestamp = []

    chunk_start_time = 0 
    full_transcript = ""

    last_chunk_processed = False

    for i in range(num_batches + 1):
        if last_chunk_processed:
            break
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(speech))  # Handle the last chunk which might be smaller
        next_chunk_end_idx = min((i + 2) * batch_size, len(speech)) 
        next_chunk_duration = (next_chunk_end_idx - end_idx) * 1.0 / sample_rate
        # chech the duration of the last chunk, if it is smaller than 3 seconds merge it with the previous
        if next_chunk_end_idx > end_idx and next_chunk_duration <= 3.0:
            end_idx = next_chunk_end_idx
            last_chunk_processed = True


        audio_chunk = speech[start_idx:end_idx].unsqueeze(0) # to make the tensor suitable for silero model
        chunk_duration = len(audio_chunk[0]) * 1.0 / sample_rate
        transcript_chunk = model(audio_chunk)

        cnt = 0
        for vector in transcript_chunk:
            
            words_timestamp = decoder(vector.cpu(), chunk_duration, word_align=True)
            # print(words_timestamp)
            if words_timestamp == "": # empty chunk
                logger.info(f"Found empty chunk starting from {chunk_start_time}. The model just ingored it")
                continue

            for word_dict in words_timestamp[-1]:
                word = word_dict['word']
                full_transcript += word + " "
                transcript_with_timestamp.append({'word': word, 
                                                  'start': chunk_start_time + word_dict['start_ts'],
                                                  'end': chunk_start_time + word_dict['end_ts'],})
            cnt += 1

        chunk_start_time += chunk_duration
    # print(full_transcript)
    transcript_with_timestamp.append({'full_transcript': full_transcript})
    return transcript_with_timestamp


def transribe_with_whisper(whisper_model, ogg_file):
    transcribe_result = whisper_model.transcribe(ogg_file, word_timestamps=True)
    return transcribe_result


def transcribe_file(audio_file, transcribtion_model, **kwargs):


    start_time = time.time()

    if type(transcribtion_model) is vosk.Model: # transcribe with vosk
        transcribe_result = transcribe_with_vosk(transcribtion_model, audio_file)

    elif type(transcribtion_model) is Wav2Vec2ForCTC:  # transcribe with Wav2Vec
        transcribe_result =  transribe_with_wav2vec(transcribtion_model, audio_file, tokenizer=kwargs.get("tokenizer"))

    elif type(transcribtion_model) is whisper.model.Whisper: # whisper model
        transcribe_result =  transribe_with_whisper(transcribtion_model, audio_file)
    
    elif type(transcribtion_model) is torch.jit._script.RecursiveScriptModule:  # transcribe with Silero models
        transcribe_result =  transcribe_with_silero(transcribtion_model, audio_wav=audio_file, 
                                                    decoder=kwargs.get("decoder"),
                                                    prepare_model_input=kwargs.get("prepare_model_input"),
                                                    read_batch=kwargs.get("read_batch"),
                                                    logger=kwargs.get("logger"),)
    else:
        raise ValueError("Transcription model not in the known list")
        

    exec_time = time.time() - start_time
    return transcribe_result, exec_time


def get_wav_file(ogg_file):
    # convert the ogg file to wav format
    wav_file = ogg_file.split(".")[0]+".wav" # path to save the converted file
    if not os.path.exists(wav_file): # if the wav file is not there, convert the ogg file to wav format

        if get_file_size(ogg_file) > 100: # use ffmpeg method for files bigger than 100 MB
            convert_ogg_to_wav(ogg_file, wav_file, method='ffmpeg')
        else:            
            convert_ogg_to_wav(ogg_file, wav_file, method='pydub')
    return wav_file
    

def transcribe_input(input_file, logger, transcribe_model, time_log, save_dir, **kwargs):
    
    # Create the output directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    missed = [] # files that were not transcribed correctly
    df = pd.read_csv(input_file)    
    for ogg_file in df['episode_path']:
        
        wav_file = get_wav_file(ogg_file)
        audio_file = ogg_file if type(transcribe_model) is whisper.model.Whisper else wav_file
        # save_file = os.path.splitext(audio_file)[0] + ".json" # to save it in the same directory but with json extension
        file_name = os.path.splitext(os.path.basename(audio_file))[0] + ".json" # Extract filename without extension, then add ".json" extension
        save_file = os.path.join(save_dir, file_name)

        if os.path.exists(save_file): # if the transcribtion file is there, no need to transcribe it again
            logger.info(f"already processed episode at {audio_file} is . Skipping it")
            continue

        logger.info(f"Transcribing the podcast from path {audio_file}")
        transcribe_result, exec_time = transcribe_file(audio_file, transcribe_model, logger=logger, **kwargs)

        if transcribe_result is None:
            missed.append(audio_file)
            logger.info(f"Could not transcribe {audio_file}")
            continue


        with open(save_file, 'w') as file:
            file.write(json.dumps(transcribe_result))  # Write the JSON line to the file

        # save the execution time & file size
        with open(time_log, 'a') as file:
            file.write(json.dumps({audio_file: exec_time, "size": os.path.getsize(audio_file)})+"\n")  # Write the JSON line to the file

        logger.info("Done saving the transcript")

    df_missed = pd.DataFrame()
    df_missed['missed'] = missed
    df_missed.to_csv(f"{save_dir}/missed_epsiodes_of_{os.path.basename(input_file)}.csv", index=False)





def transcribe_directory(directory = "/storage/users/watheq/projects/podcast_search/data/dataset_files/audio_sample/sample"):
    '''
    For transcribing all audio files within a given directory
    '''
    base_model = whisper.load_model("base", device=device)
    time_file = "/storage/users/watheq/projects/podcast_search/data/dataset_files/audio_sample/time_cost.txt"
    log_file = "/storage/users/watheq/projects/podcast_search/data/dataset_files/audio_sample/transcribe_sample.log"
    logger = get_logger(log_file)

    # Iterate over all files in the directory and subdirectories, and transcribe the audio files
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if file_path.endswith(".ogg"):
                transcribe_file(file_path, logger, transcribtion_model=base_model, time_log=time_file)
        # print(file_path)




def main():
    parser = argparse.ArgumentParser(description="Script to transcribe a list of audio files saved in a csv file")
    parser.add_argument("--input", type=str, required=True, help="Path to the input file that contains a list of paths to the audio files")
    parser.add_argument("--model", type=str, required=True, help="The transcription model of Whisper (small, base, medium, or large)")
    parser.add_argument("--log", type=str, required=True, help="Path to save the execution log")
    parser.add_argument("--time_log", type=str, required=True, help="Filepath to save the time log for each audio file")
    parser.add_argument("--output", type=str, required=True, help="Directory to save the transcribed files")
    args = parser.parse_args()
    input = args.input
    model = args.model
    log_file = args.log
    time_log = args.time_log
    output_dir = args.output

    # input = '/storage/users/watheq/projects/podcast_search/transcription/reranking_episodes/part_1.csv'
    # model = 'vosk-small'

    logger = get_logger(log_file)
    logger.info(f"Working on device  {device}")
    logger.info(f"The input is read from {input}")
    logger.info(f"The log_file will be saved to {log_file}")
    logger.info(f"The time_log will be saved to {time_log}")
    logger.info(f"The transcribed files will be saved to {output_dir}")

    tokenizer = None
    silero_decoder = None
    silero_prepare_model_input = None
    silero_read_batch = None

    if model.startswith('vosk'):
        # vosk models:  https://alphacephei.com/vosk/models
        if 'small' in model:
            model_name = "vosk-model-small-en-us-0.15" # 40M, Lightweight wideband model for Android and RPi, Word error rate (WER): 9.85 (librispeech test-clean) 10.38 (tedlium); 
        elif 'large' in model:
            model_name = "vosk-model-en-us-0.42-gigaspeech" # 2.3G, Accurate generic US English model trained by Kaldi on Gigaspeech. Mostly for podcasts, not for telephony
            # WER: 5.64 (librispeech test-clean) 6.24 (tedlium) 30.17 (callcenter) 	

        transcribtion_model = load_model(model_name=model_name)
    
    elif model.startswith('wav2vec'):
        model_name = f"facebook/{model}"
        tokenizer = Wav2Vec2Tokenizer.from_pretrained(model_name)
        # processor = Wav2Vec2Processor.from_pretrained(model_name)
        transcribtion_model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device)

        
    elif model.startswith("whisper"):
        transcribtion_model = whisper.load_model(model, device=device)

    elif model.startswith("silero"):
        if model.find('small') != -1:
            jit_model = 'jit'
        elif model.find("large") != -1:
            jit_model = 'jit_xlarge'

        transcribtion_model, silero_decoder, silero_utils = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                       model='silero_stt',
                                       version='v6',
                                       jit_model=jit_model, # jit (111 MB), jit_xlarge (527 MB)
                                       language='en', # also available 'de', 'es'
                                       device=device)
        (silero_read_batch, split_into_batches,
        read_audio, silero_prepare_model_input) = silero_utils 

        logger.info(f"Silero model was loaded size {jit_model}")


    transcribe_input(input, logger, transcribtion_model, time_log, save_dir=output_dir,
                     # **kwargs
                    tokenizer=tokenizer,decoder=silero_decoder, 
                    prepare_model_input=silero_prepare_model_input,
                    read_batch=silero_read_batch)




if __name__ == "__main__":
    main()


