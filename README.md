# Modality-Translation-Learning-for-Joint-Speech-Text-Model

## [Steps to install fairseq](https://github.com/facebookresearch/fairseq/issues/5289#issuecomment-1906697931)

CAUTION: always use `python -m pip` instead of `pip` otherwise it will install globally instead of just installing inside the environment.

1. create and activate a conda environment with python version 3.9
   for example:

   ```
   conda create --name myEnv python=3.9
   conda activate myEnv
   ```

2. install Pytorch v1.10.1 with GPU support:

```
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
```

3. clone fairseq repository and install editable:

```
git submodule update --init fairseq
cd fairseq
python -m pip install --editable ./
python setup.py build develop
```

4. upgrade hydra-core and numpy and tensorboard:

```
python -m pip install --upgrade hydra-core numpy tensorboard
```

Ignore any errors and warnings on dependencies after running this. ( warnings related to omegaconf will show up, but fairseq will work fine)

5. install flashlight python bindings:

```
python -m pip install flashlight-text flashlight-sequence
```

6. install sacrebleu:

```
python -m pip install sacrebleu==1.5.1
```

## Data Preparation

### LibriSpeech

- Follow the steps of wav2vec 2.0 manifest [here](https://github.com/pytorch/fairseq/tree/main/examples/wav2vec#prepare-training-data-manifest) to prepare `train.tsv` and `train.ltr`.
  :::info
  tsv: `wav2vec_manifest.py` ([wav2vec_manifest.py](https://github.com/facebookresearch/fairseq/blob/main/examples/wav2vec/wav2vec_manifest.py))
  ltr: `libri_labels.py` ([libri_labels.py](https://github.com/facebookresearch/fairseq/blob/main/examples/wav2vec/libri_labels.py))
  :::
  - You should make sure the vocabulary [dict.ltr.txt](https://github.com/microsoft/SpeechT5/blob/main/SpeechLM/dataset/LibriSpeech/asr/dict.ltr.txt) is the same as that used for the pre-trained model.
  - Put yout prepared data into `$data_dir`
- Prepare validation data `dev-clean`

```
python examples/wav2vec/wav2vec_manifest.py /home/espnet/egs2/librispeech/asr1/downloads/LibriSpeech/dev-clean --dest /home/SpeechT5/SpeechLM/dataset/LibriSpeech/asr --ext flac --valid-percent 0
```

### IEMOCAP

- manifest.py to get tsv

```
python examples/wav2vec/wav2vec_manifest.py \
/home/espnet/egs2/iemocap/asr1/download/IEMOCAP_full_release/Session1/sentences/wav \
--dest /home/SpeechT5/SpeechLM/dataset/iemocap/asr --ext wav --valid-percent 0
```

- labels.py to get ltr

```
python iemocap_labels.py \
/home/SpeechT5/SpeechLM/dataset/iemocap/asr/train.tsv \
--output-dir /home/SpeechT5/SpeechLM/dataset/iemocap/asr \
--output-name session1
```

## Pre-train

### Prepare the pre-training data (SpeechLM)

#### [Tokenizers](https://github.com/microsoft/SpeechT5/tree/main/SpeechLM#tokenizers)

- Phoneme-unit Tokenizer for Text
  - This tokenizer is used to phonemize the unpaired text data to (phonemes, letters) paired data, following a `words -> phonemes -> upsampled phones` pipeline
  - The following script will download LibriSpeech LM corpus and produce the required data: `train_text.phn-ltr.phn.{idx,bin}` and `train_text.phn-ltr.ltr.{idx,bin}`

```
# data will be in dataset/LibriLM/phone_unit/
bash speechlm/data_process/prepare_phn2ltr_librilm.sh
```

- Hidden-unit Tokenizer for Speech (follow [HuBERT](https://github.com/facebookresearch/fairseq/tree/main/examples/hubert#data-preparation))
  - Prepare 1) wav recodings `train.tsv` and 2) corresponding hidden-units `train.km`, and 3) unit vocabulary `dict.km.txt`
  - Use `create_km.sh` in `/home/SpeechT5/SpeechLM/fairseq/examples/hubert/simple_kmeans` to prepare these data
- Hidden-unit Tokenizer for Text

  - produce the speech-style hidden units from unpaired text
  - Train:
    1. Convert asr transcripts to phoneme sequence with duration information
    2. Extract hidden-units from speech, using the Hidden-unit Tokenizer for Speech
    3. Train the model (fasttext2unit) on the paired data:
    ```bash
    data_dir=dataset/LibriSpeech/fast_phone2unit
    bash speechlm/scripts/tokenizer_fastT2U/train_s_5e-4.sh $data_dir
    ```
  - Inference: 4. Convert text data to phoneme sequence by [lexicon](https://drive.google.com/file/d/1dh9NEx_cCF9_Aa0UcKyl9j00GXs6LmLQ/view) 5. Generate hidden units for a large text corpus:
    ```bash
    gen_set=dataset/LibriSpeech/fast_phone2unit/genset_examples
    bash speechlm/scripts/tokenizer_fastT2U/generate.sh $model_path $gen_set
    ```

- Here is the step for generating paired pre-training unit data (Text) 1. Prepare text data (tsv) 2. Align the order of text data with speech tsv `/home_new/SpeechT5/SpeechLM/speechlm/data_process/align_text_order_with_tsv.py` 3. `SpeechT5/SpeechLM/speechlm/data_process/prepare_phn2ltr_librispeech100.sh` for preparing phoneme data (phn) 4. `SpeechT5/SpeechLM/speechlm/scripts/tokenizer_fastT2U/generate.sh` for generating unit data
  :::info
  You can directly use `prepare_phn2ltr_librispeech.sh`
  :::

#### [Pre-train](https://github.com/microsoft/SpeechT5/tree/main/SpeechLM#pre-train)

- SpeechLM-H Base model

```bash
data_dir=dataset/LibriSpeech/hidden_unit  # should contain train_960.{tsv,phn}
text_data_dir=dataset/LibriLM/km-ltr/bin-idx     # should contain train_text.km-ltr.{km,ltr}.{bin,idx}
# Usage: speechlm/scripts/pretrain_speechlm/base_speechlmh.sh <data_dir> <text_data_dir> [mount=$PWD] [world_size=32] [update_freq=1]
bash speechlm/scripts/pretrain_speechlm/base_speechlmh.sh $data_dir $text_data_dir
```

## Fine-tune

### wav2vec 2.0

#### Prepare training data manifest

- **training data**

```
python examples/wav2vec/wav2vec_manifest.py examples/wav2vec/data --dest examples/wav2vec/manifest --ext flac --valid-percent 0
```

$ext should be set to flac, wav, or whatever format your dataset happens to use that soundfile can read.

$valid should be set to some reasonable percentage (like 0.01) of training data to use for validation. To use a pre-defined validation set (like dev-other from librispeech), set to it 0 and then overwrite valid.tsv with a separately pre-processed manifest file.

- **validation data**

```
python examples/wav2vec/wav2vec_manifest.py examples/wav2vec/data/validation --dest examples/wav2vec/manifest/validation --ext flac --valid-percent 0
```

#### Train a wav2vec 2.0 base model

```
fairseq-hydra-train \
    task.data=/home/fairseq/examples/wav2vec/manifest \
    distributed_training.distributed_world_size=1 +optimization.update_freq='[64]' \
    --config-dir examples/wav2vec/config/pretraining \
    --config-name wav2vec2_base_librispeech
```

#### Fine-tune a pre-trained model with CTC

Fine-tuning a model requires parallel audio and labels file, as well as a vocabulary file in fairseq format. A letter vocabulary can be downloaded [here](https://dl.fbaipublicfiles.com/fairseq/wav2vec/dict.ltr.txt). An example script that generates labels for the Librispeech dataset from the tsv file produced by wav2vec_manifest.py can be used as follows:

config: base_1h.yaml

build labels for training data

```
python libri_labels.py /home/fairseq/examples/wav2vec/manifest/train.tsv --output-dir /home/fairseq/examples/wav2vec/manifest --output-name train
```

build labels for validation data(dev_other)

```
python libri_labels.py /home/fairseq/examples/wav2vec/manifest/dev_other.tsv --output-dir /home/fairseq/examples/wav2vec/manifest --output-name dev_other
```

Prepare validation data manifest

```
python wav2vec_manifest.py data/validation/dev-other --dest manifest --ext flac --valid-percent 0
```

Fine-tuning on 100h of Librispeech with letter targets:

```


$ fairseq-hydra-train \
    task.data=/home/fairseq/examples/wav2vec/manifest \
    model.w2v_path=/home/fairseq/examples/wav2vec/wav2vec_small.pt \
    distributed_training.distributed_world_size=1 \
    +optimization.update_freq='[24]' \
    --config-dir /home/fairseq/examples/wav2vec/config/finetuning \
    --config-name base_1h
```

#### Inference

`bash /home_new/fairseq/examples/wav2vec/inference_ctc.sh`

**If you face this problem** [ModuleNotFoundError: No module named 'examples.speech_to_text'](https://github.com/facebookresearch/fairseq/issues/3087) **, please follow** [Steps to install fairseq](#Steps-to-install-fairseq)

### SpeechLM

#### LibriSpeech

- Fine-tune the base model
  ```
  bash speechlm/scripts/tune_speechlm_asr/finetune_base_ctc.sh \
  /home/SpeechT5/SpeechLM/models/speechlmh_base_checkpoint_298_400000.pt \
  /home/SpeechT5/SpeechLM/dataset/LibriSpeech/asr 'tag400k'
  ```
- Solve the problem that model try to access no-existing config files when fine-tuning
  [reference link](https://github.com/microsoft/SpeechT5/commit/e2be6fa52ed8214464f7f7e94543dc82dd535e85)
  [reference link](https://github.com/microsoft/SpeechT5/issues/34)
- Numpy float
  ```
  pip uninstall numpy
  pip install numpy==1.23.0
  ```

#### IMOCAP

- Fine-tune the base model
  ```
  bash speechlm/scripts/tune_speechlm_asr/finetune_base_ctc.sh \
  /home/SpeechT5/SpeechLM/models/speechlmh_base_checkpoint_298_400000.pt \
  /home/SpeechT5/SpeechLM/dataset/iemocap/asr 'iemocap'
  ```
  ```
  tensorboard --logdir /path/to/modeldir
  ```
- Lable.ltr 要把所有字母轉成大寫
