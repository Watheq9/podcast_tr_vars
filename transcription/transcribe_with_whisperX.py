import os, sys
sys.path.insert(0, os.getcwd())
import time
import os
import numpy as np
import torch
import pandas as pd
import json
import logging
import argparse
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2Tokenizer
import whisperx
import configure as cf
import gc

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


def get_file_size(file_path):
    stat_info = os.stat(file_path)
    file_size_mb = stat_info.st_size / (1024 * 1024)
    return file_size_mb



def transcribe_with_whisperX(model, audio_wav, alignment_model, diarize_model, 
                             alignment_metadata, language='en', batch_size = 16, diarize=False):

    audio = whisperx.load_audio(audio_wav)

    # 1. Transcribe with original whisper (batched)
    result = model.transcribe(audio, batch_size=batch_size, language=language)

    # 2. Align whisper output
    transcript = whisperx.align(result["segments"], alignment_model, alignment_metadata, audio, device, return_char_alignments=False)

    # free memory
    del result 
    del audio 
    gc.collect()


    if not diarize:
        return transcript
    
    # 3. Assign speaker labels
    # add min/max number of speakers if known
    diarize_segments = diarize_model(audio)
    # diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)

    transcript = whisperx.assign_word_speakers(diarize_segments, transcript)

    return transcript


def transcribe_file(audio_file, transcribtion_model, **kwargs):


    start_time = time.time()

    transcribe_result =  transcribe_with_whisperX(transcribtion_model, audio_wav=audio_file, **kwargs)
                                                # alignment_model=kwargs.get("alignment_model"),
                                                # diarize_model=kwargs.get("diarize_model"),
                                                # alignment_metadata=kwargs.get("alignment_metadata"),
                                                # language=kwargs.get("language"),
                                                # batch_size=kwargs.get("batch_size"),
                                                # diarize=kwargs.get("diarize"),)
            


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
    skipped = []
    for ogg_file in df['episode_path']:
        
        audio_file = ogg_file
        file_name = os.path.splitext(os.path.basename(audio_file))[0] + ".json" # Extract filename without extension, then add ".json" extension
        save_file = os.path.join(save_dir, file_name)

        if os.path.exists(save_file): # if the transcribtion file is there, no need to transcribe it again
            skipped.append(audio_file)
            continue

        logger.info(f"Transcribing the podcast from path {audio_file}")
        transcribe_result, exec_time = transcribe_file(audio_file, transcribe_model, **kwargs)

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

    logger.info(f"Done transcribing. Number of already processed episodes is {len(skipped)}. They are {skipped}")

def parse_boolean(arg):
    ua = str(arg).upper()
    print(ua)
    if 'TRUE'.startswith(ua):
       return True
    elif 'FALSE'.startswith(ua):
        print("Entered false")
        return False
        
    else:
       raise KeyError("Error in the argument value")


import ast

def main():
    parser = argparse.ArgumentParser(description="Script to transcribe a list of audio files saved in a csv file")
    parser.add_argument("--input", type=str, required=True, help="Path to the input file that contains a list of paths to the audio files")
    parser.add_argument("--model", type=str, required=True, help="The transcription model of Whisper (small, base, medium, or large)")
    parser.add_argument("--log", type=str, required=True, help="Path to save the execution log")
    parser.add_argument("--time_log", type=str, required=True, help="Filepath to save the time log for each audio file")
    parser.add_argument("--lang", type=str, default='en', required=False, help="The language of input audio files")
    parser.add_argument("--ctype", type=str, default='float16', required=False, help="Compute type for whisper x. Can be float16, int8 , etc.")
    parser.add_argument("--batch", type=int, default=16, required=False, help="Batch size")
    parser.add_argument("--diarize", type=bool, default=False, required=False, help="Reconginize the speakers in the audio file")
    parser.add_argument("--output", type=str, required=True, help="Directory to save the transcribed files")
    args = parser.parse_args()
    input = args.input
    model = args.model
    log_file = args.log
    time_log = args.time_log
    output_dir = args.output
    lang = args.lang
    diarize = args.diarize
    batch = args.batch
    compute_type =  args.ctype # change to "int8" if low on GPU mem (may reduce accuracy)

    logger = get_logger(log_file)
    logger.info(f"Working on device  {device}")
    logger.info(f"The input is read from {input} with language {lang}, compute type {compute_type}, and speaker diarization set to {diarize} ")
    logger.info(f"The log_file will be saved to {log_file}")
    logger.info(f"The time_log will be saved to {time_log}")
    logger.info(f"The transcribed files will be saved to {output_dir}")


    if model.startswith('whisperX'):
        # vosk models:  https://alphacephei.com/vosk/models
        whisper_model = model.split("whisperX")[1][1:]
        transcribtion_model = whisperx.load_model(whisper_model, device, compute_type=compute_type,
                                                      language=lang, download_root=cf.models_dir)

    else:
        raise ValueError("Transcription model not in the known list")

    alignment_model, alignment_metadata = whisperx.load_align_model(language_code=lang, device=device)
    diarize_model = None # no need to load the model if no diarization
    if diarize:
        diarize_model = whisperx.DiarizationPipeline(use_auth_token=cf.HF_TOKEN, device=device)
        logger.info("loaded the diarization model successfully")

    logger.info(f"WhisperX model was loaded successuflly with type {model} and batch size {batch}")


    transcribe_input(input, logger, transcribtion_model, time_log, save_dir=output_dir,
                     # **kwargs
                    alignment_model=alignment_model,diarize_model=diarize_model, 
                    alignment_metadata=alignment_metadata, language=lang, batch_size = batch, diarize=diarize) 



if __name__ == "__main__":
    main()


'''
====================================== WhisperX-large-v3 =================================================


model='whisperX-large-v3'
part=1
experiment=transcribe_reranking_set_p${part}
python transcription/02_transcribe_with_whisperX.py \
        --input /storage/users/watheq/projects/podcast_search/transcription/reranking_episodes/part_${part}.csv \
        --model ${model} \
        --log /storage/users/watheq/projects/podcast_search/data/logs/${experiment}_with_${model}.log \
        --time_log /storage/users/watheq/projects/podcast_search/data/logs/${experiment}_with_${model}_time_.log \
        --output  /storage/users/watheq/projects/podcast_search/data/reranking_transcribe/${model}

        
model='whisperX-large-v3'
part=2
experiment=transcribe_reranking_set_p${part}
CUDA_VISIBLE_DEVICES=1  python transcription/02_transcribe_with_whisperX.py \
        --input /storage/users/watheq/projects/podcast_search/transcription/reranking_episodes/part_${part}.csv \
        --model ${model} \
        --log /storage/users/watheq/projects/podcast_search/data/logs/${experiment}_with_${model}.log \
        --time_log /storage/users/watheq/projects/podcast_search/data/logs/${experiment}_with_${model}_time_.log \
        --output  /storage/users/watheq/projects/podcast_search/data/reranking_transcribe/${model}



          
====================================== WhisperX-base =================================================


model='whisperX-base'
part=1
experiment=transcribe_reranking_set_p${part}
python transcription/02_transcribe_with_whisperX.py \
        --input /storage/users/watheq/projects/podcast_search/transcription/reranking_episodes/part_${part}.csv \
        --model ${model} \
        --log /storage/users/watheq/projects/podcast_search/data/logs/${experiment}_with_${model}.log \
        --time_log /storage/users/watheq/projects/podcast_search/data/logs/${experiment}_with_${model}_time_.log \
        --output  /storage/users/watheq/projects/podcast_search/data/reranking_transcribe/${model}

        
model='whisperX-base'
part=2
experiment=transcribe_reranking_set_p${part}
CUDA_VISIBLE_DEVICES=1  python transcription/02_transcribe_with_whisperX.py \
        --input /storage/users/watheq/projects/podcast_search/transcription/reranking_episodes/part_${part}.csv \
        --model ${model} \
        --log /storage/users/watheq/projects/podcast_search/data/logs/${experiment}_with_${model}.log \
        --time_log /storage/users/watheq/projects/podcast_search/data/logs/${experiment}_with_${model}_time_.log \
        --output  /storage/users/watheq/projects/podcast_search/data/reranking_transcribe/${model}





====================================== WhisperX-base =========== whole corpus ======================================

## ---------------------- Step 1 : generate transcripts with speaker identification --------------------

model='whisperX-base'
part=1
diarize=True
experiment=transcribe_whole_corpus_part${part}_using_${model}_with_diarization
CUDA_VISIBLE_DEVICES=1  python transcription/02_transcribe_with_whisperX.py \
        --input /storage/users/watheq/projects/podcast_search/transcription/full_corpus_episodes/part_${part}.csv \
        --model ${model} \
        --log /storage/users/watheq/projects/podcast_search/data/logs/${experiment}.log \
        --time_log /storage/users/watheq/projects/podcast_search/data/logs/time_of_${experiment}.log \
        --diarize ${diarize} \
        --output  /storage/users/watheq/projects/podcast_search/data/corpus/${model}


## ---------------------- Step 1 : generate transcripts without speaker identification --------------------
model='whisperX-base'
part=1
diarize=False
experiment=transcribe_whole_corpus_part${part}_using_${model}_without_diarization
CUDA_VISIBLE_DEVICES=1  python transcription/02_transcribe_with_whisperX.py \
        --input /storage/users/watheq/projects/podcast_search/transcription/full_corpus_episodes/part_${part}.csv \
        --model ${model} \
        --log /storage/users/watheq/projects/podcast_search/data/logs/${experiment}.log \
        --time_log /storage/users/watheq/projects/podcast_search/data/logs/time_of_${experiment}.log \
        --output  /storage/users/watheq/projects/podcast_search/data/corpus/${model}


model='whisperX-base'
part=2
diarize=False
experiment=transcribe_whole_corpus_part${part}_using_${model}_without_diarization
python transcription/02_transcribe_with_whisperX.py \
        --input /storage/users/watheq/projects/podcast_search/transcription/full_corpus_episodes/part_${part}.csv \
        --model ${model} \
        --log /storage/users/watheq/projects/podcast_search/data/logs/${experiment}.log \
        --time_log /storage/users/watheq/projects/podcast_search/data/logs/time_of_${experiment}.log \
        --output  /storage/users/watheq/projects/podcast_search/data/corpus/${model}

model='whisperX-base'
part=3
diarize=False
experiment=transcribe_whole_corpus_part${part}_using_${model}_without_diarization
python transcription/02_transcribe_with_whisperX.py \
        --input /storage/users/watheq/projects/podcast_search/transcription/full_corpus_episodes/part_${part}.csv \
        --model ${model} \
        --log /storage/users/watheq/projects/podcast_search/data/logs/${experiment}.log \
        --time_log /storage/users/watheq/projects/podcast_search/data/logs/time_of_${experiment}.log \
        --output  /storage/users/watheq/projects/podcast_search/data/corpus/${model}

model='whisperX-base'
part=4
diarize=False
experiment=transcribe_whole_corpus_part${part}_using_${model}_without_diarization
python transcription/02_transcribe_with_whisperX.py \
        --input /storage/users/watheq/projects/podcast_search/transcription/full_corpus_episodes/part_${part}.csv \
        --model ${model} \
        --log /storage/users/watheq/projects/podcast_search/data/logs/${experiment}.log \
        --time_log /storage/users/watheq/projects/podcast_search/data/logs/time_of_${experiment}.log \
        --output  /storage/users/watheq/projects/podcast_search/data/corpus/${model}

model='whisperX-base'
part=5
diarize=False
experiment=transcribe_whole_corpus_part${part}_using_${model}_without_diarization
python transcription/02_transcribe_with_whisperX.py \
        --input /storage/users/watheq/projects/podcast_search/transcription/full_corpus_episodes/part_${part}.csv \
        --model ${model} \
        --log /storage/users/watheq/projects/podcast_search/data/logs/${experiment}.log \
        --time_log /storage/users/watheq/projects/podcast_search/data/logs/time_of_${experiment}.log \
        --output  /storage/users/watheq/projects/podcast_search/data/corpus/${model}

        
        
## ---------------------- Step 2 : form 2-minute segments using the generated transcripts -------------------
model='whisperX-base'
transcript_dir="/storage/users/watheq/projects/podcast_search/data/corpus/${model}"
seg_length=120
seg_step=60
python helper/generate_segments.py \
    --log_file "/storage/users/watheq/projects/podcast_search/data/logs/${model}_${seg_length}_${seg_step}_time_segment.txt" \
    --save_file "/storage/users/watheq/projects/podcast_search/data/dataset_files/${model}_${seg_length}_${seg_step}_time_segment.jsonl" \
    --seg_length ${seg_length} \
    --seg_step ${seg_step} \
    --seg_type time \
    --model ${model} \
    --transcript_dir ${transcript_dir}

    
## ---------------------- Step 2 : form 2-minute segments with extra 5 seconds on both ends of each each segment -------------------
model='whisperX-base'
transcript_dir="/storage/users/watheq/projects/podcast_search/data/corpus/${model}"
seg_length=120
seg_step=60
extra_sec=5
python helper/generate_segments.py \
    --log_file "/storage/users/watheq/projects/podcast_search/data/logs/${model}_${seg_length}_${seg_step}_time_segment_extra_${extra_sec}.txt" \
    --save_file "/storage/users/watheq/projects/podcast_search/data/dataset_files/${model}_${seg_length}_${seg_step}_extra_${extra_sec}_time_segment.jsonl" \
    --seg_length ${seg_length} \
    --seg_step ${seg_step} \
    --seg_type time \
    --model ${model} \
    --extra_sec ${extra_sec} \
    --transcript_dir ${transcript_dir}
    
## ---------------------- Step 3 : Index the segments using pyterrier -------------------

------------one-field-------
model='whisperX-base'
seg_length=120
seg_step=60
type='one-field'
corpus="/storage/users/watheq/projects/podcast_search/data/dataset_files/${model}_${seg_length}_${seg_step}_time_segment.jsonl"
index_dir="/storage/users/watheq/projects/podcast_search/data/indexes/terrier/${model}_${seg_length}_${seg_step}_${type}_time_segment"
python helper/index.py \
    --log_file /storage/users/watheq/projects/podcast_search/data/logs/Indexing_${model}_${seg_length}_${seg_step}_time_segment_${type}.txt \
    --corpus ${corpus} \
    --index_dir ${index_dir} \
    --fields seg_words  \
    --index_type terrier-one-field \
    --meta_fields minutes_ids


------------one-field--extra 5 sec-----
model='whisperX-base'
seg_length=120
seg_step=60
type='one-field'
extra_sec=5
corpus="/storage/users/watheq/projects/podcast_search/data/dataset_files/${model}_${seg_length}_${seg_step}_extra_${extra_sec}_time_segment.jsonl" 
index_dir="/storage/users/watheq/projects/podcast_search/data/indexes/terrier/${model}_${seg_length}_${seg_step}_${type}_extra_${extra_sec}_time_segment"
python helper/index.py \
    --log_file /storage/users/watheq/projects/podcast_search/data/logs/Indexing_${model}_${seg_length}_${seg_step}_time_segment_${type}.txt \
    --corpus ${corpus} \
    --index_dir ${index_dir} \
    --fields seg_words  \
    --index_type terrier-one-field \
    --meta_fields minutes_ids


------------multi-field-------
model='whisperX-base'
seg_length=120
seg_step=60
type='multi-field'
corpus="/storage/users/watheq/projects/podcast_search/data/dataset_files/${model}_${seg_length}_${seg_step}_time_segment.jsonl"
index_dir="/storage/users/watheq/projects/podcast_search/data/indexes/terrier/${model}_${seg_length}_${seg_step}_${type}_time_segment"
python helper/index.py \
    --log_file /storage/users/watheq/projects/podcast_search/data/logs/Indexing_${model}_${seg_length}_${seg_step}_time_segment_${type}.txt \
    --corpus ${corpus} \
    --index_dir ${index_dir} \
    --fields seg_words show_name show_desc epis_name epis_desc \
    --index_type terrier-multi-field \
    --meta_fields minutes_ids


==================================WhisperX-large =============================
## ---------------------- Step 1 : generate transcripts without speaker identification --------------------
model='whisperX-large-v3'
part=1
diarize=False
experiment=transcribe_whole_corpus_part${part}_using_${model}_without_diarization
CUDA_VISIBLE_DEVICES=1  python transcription/02_transcribe_with_whisperX.py \
        --input /storage/users/watheq/projects/podcast_search/transcription/full_corpus_episodes/part_${part}.csv \
        --model ${model} \
        --log /storage/users/watheq/projects/podcast_search/data/logs/${experiment}.log \
        --time_log /storage/users/watheq/projects/podcast_search/data/logs/time_of_${experiment}.log \
        --output  /storage/users/watheq/projects/podcast_search/data/corpus/${model}


model='whisperX-large-v3'
part=2
diarize=False
experiment=transcribe_whole_corpus_part${part}_using_${model}_without_diarization
python transcription/02_transcribe_with_whisperX.py \
        --input /storage/users/watheq/projects/podcast_search/transcription/full_corpus_episodes/part_${part}.csv \
        --model ${model} \
        --log /storage/users/watheq/projects/podcast_search/data/logs/${experiment}.log \
        --time_log /storage/users/watheq/projects/podcast_search/data/logs/time_of_${experiment}.log \
        --output  /storage/users/watheq/projects/podcast_search/data/corpus/${model}




model='whisperX-large-v3'
part=3
diarize=False
experiment=transcribe_whole_corpus_part${part}_using_${model}_without_diarization
python transcription/02_transcribe_with_whisperX.py \
        --input /storage/users/watheq/projects/podcast_search/transcription/full_corpus_episodes/part_${part}.csv \
        --model ${model} \
        --log /storage/users/watheq/projects/podcast_search/data/logs/${experiment}.log \
        --time_log /storage/users/watheq/projects/podcast_search/data/logs/time_of_${experiment}.log \
        --output  /storage/users/watheq/projects/podcast_search/data/corpus/${model}

        


## ---------------------- Step 2 : form 2-minute segments using the generated transcripts -------------------
model='whisperX-large-v3'
transcript_dir="/storage/users/watheq/projects/podcast_search/data/corpus/${model}"
seg_length=120
seg_step=60
python helper/generate_segments.py \
    --log_file "/storage/users/watheq/projects/podcast_search/data/logs/${model}_${seg_length}_${seg_step}_time_segment.txt" \
    --save_file "/storage/users/watheq/projects/podcast_search/data/dataset_files/${model}_${seg_length}_${seg_step}_time_segment.jsonl" \
    --seg_length ${seg_length} \
    --seg_step ${seg_step} \
    --seg_type time \
    --model ${model} \
    --transcript_dir ${transcript_dir}

    

    
## ---------------------- Step 3 : Index the segments using pyterrier -------------------

------------one-field-------
model='whisperX-large-v3'
seg_length=120
seg_step=60
type='one-field'
corpus="/storage/users/watheq/projects/podcast_search/data/dataset_files/${model}_${seg_length}_${seg_step}_time_segment.jsonl"
index_dir="/storage/users/watheq/projects/podcast_search/data/indexes/terrier/${model}_${seg_length}_${seg_step}_${type}_time_segment"
python helper/index.py \
    --log_file /storage/users/watheq/projects/podcast_search/data/logs/Indexing_${model}_${seg_length}_${seg_step}_time_segment_${type}.txt \
    --corpus ${corpus} \
    --index_dir ${index_dir} \
    --fields seg_words  \
    --index_type terrier-one-field \
    --meta_fields minutes_ids
'''