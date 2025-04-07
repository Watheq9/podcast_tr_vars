#!/bin/bash

## ---------------------- Step 1 : generate transcripts --------------------
project_dir="set_your_project_directory"
cd $project_dir


# 1. Transcribe with Silero models 
tr_models=(\
        "silero-small" \
        "silero-large" \
        "vosk-small" \
        "vosk-large" \
        "wav2vec2-large-960h-lv60-self" \
        "whisper-tiny" \
        "whisper-base" \
        "whisper-small" \
        "whisper-medium" \
        ) #  \


for tr_model in "${tr_models[@]}"; do

    # Put the path of every single audio file in one csv file, and then pass that file to the transcription script.
    # Check episodes.csv to see a sample.
    input_file="transcription/episodes.csv"
    experiment="transcribe_whole_corpus" # you can divide the episodes file into multiple parts and run the scripts on parallel
    log_file="data/logs/${experiment}_with_${tr_model}.log"
    time_log="data/logs/time_of_${experiment}_with_${tr_model}.log"
    # directory to save the transcribed audio files
    output_dir="data/corpus/${tr_model}"
    echo "Generating transcripts using ${tr_model} using input file from ${input_file}"
    python transcription/transcribe.py \
            --input ${input_file} \
            --model ${tr_model} \
            --log ${log_file} \
            --time_log ${time_log} \
            --output  ${output_dir}

    echo "Done transcribing all files ^_^"
done



# 2. Transcribe with WhisperX models 
tr_models=(\
        "whisperX-large-v3" \
        "whisperX-base" \
        ) #  


for tr_model in "${tr_models[@]}"; do

    input_file="transcription/episodes.csv"
    experiment="transcribe_whole_corpus" 
    log_file="data/logs/${experiment}_with_${tr_model}.log"
    time_log="data/logs/time_of_${experiment}_with_${tr_model}.log"
    # directory to save the transcribed audio files
    output_dir="data/corpus/${tr_model}"
    diarize=False

    echo "Generating transcripts using ${tr_model} using input file from ${input_file}"
    CUDA_VISIBLE_DEVICES=0  python transcription/transcribe_with_whisperX.py \
        --input ${input_file} \
        --model ${tr_model} \
        --log ${log_file} \
        --time_log ${time_log} \
        --output  ${output_dir}

    echo "Done transcribing all files ^_^"
done


## ---------------------- Step 2 : form 2-minute segments using the generated transcripts -------------------
tr_models=(\
        "silero-small" \
        "silero-large" \
        "whisperX-large-v3" \
        "whisperX-base" \
        ) #  


for tr_model in "${tr_models[@]}"; do

    transcript_dir="data/corpus/${tr_model}"
    seg_length=120
    seg_step=60
    python helper/generate_segments.py \
        --log_file "${project_dir}/data/logs/${model}_${seg_length}_${seg_step}_time_segment.txt" \
        --save_file "${project_dir}/data/dataset_files/${model}_${seg_length}_${seg_step}_time_segment.jsonl" \
        --seg_length ${seg_length} \
        --seg_step ${seg_step} \
        --seg_type time \
        --model ${tr_model} \
        --transcript_dir ${transcript_dir}

done




    
## ---------------------- Step 3 : Index the segments using pyterrier -------------------

tr_models=(\
        "silero-small" \
        "silero-large" \
        "whisperX-large-v3" \
        "whisperX-base" \
        ) #  


seg_length=120
seg_step=60
type='one-field'
for tr_model in "${tr_models[@]}"; do

    corpus="${project_dir}/data/dataset_files/${model}_${seg_length}_${seg_step}_time_segment.jsonl"
    index_dir="${project_dir}/data/indexes/terrier/${model}_${seg_length}_${seg_step}_${type}_time_segment"
    python helper/index.py \
        --log_file ${project_dir}/data/logs/Indexing_${model}_${seg_length}_${seg_step}_time_segment_${type}.txt \
        --corpus ${corpus} \
        --index_dir ${index_dir} \
        --fields seg_words  \
        --index_type terrier-one-field \
        --meta_fields minutes_ids

done