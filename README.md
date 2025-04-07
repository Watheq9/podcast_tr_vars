# Examining the Impact of Transcript Variation on Podcast Search and Re-ranking 

This repository helps in reproducing our ECIR-2025 paper entitled [Examining the Impact of Transcript Variation on Podcast Search and Re-ranking](https://link.springer.com/content/pdf/10.1007%2F978-3-031-88714-7_9.pdf)



## Libraries Installation
First, create a conda environment, activate it, and then install the needed libraries. Run the following commands:

```
conda create --name PVC python=3.10
conda activate PVC
conda install -c pytorch faiss-cpu pytorch -y
pip install -r requirements.txt
```

If you do not already have JDK 21 installed, install via conda:
```
conda install -c conda-forge openjdk=21 maven -y
```


### Transcribing audio and Indexing


To generate the transcription, refer to the script **transcription/generate_and_index_transcriptions.sh**
