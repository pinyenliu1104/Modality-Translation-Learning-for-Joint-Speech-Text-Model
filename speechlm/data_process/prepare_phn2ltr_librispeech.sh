#!/bin/bash
[ ${PWD##*/} != SpeechLM ] && echo "Error: dir not match! Switch to SpeechLM/ and run it again!" && exit 1
cwd=${PWD}
src=${PWD}/speechlm/data_process

file_name=$1
dic_path=/home_new/fairseq/examples/wav2vec/data/train-clean-100
input_text=/home_new/SpeechT5/SpeechLM/dataset/LibriSpeech/asr/${file_name}.wrd
input_tsv=/home_new/SpeechT5/SpeechLM/dataset/LibriSpeech/asr/${file_name}.tsv
dest=/home_new/SpeechT5/SpeechLM/dataset/LibriSpeech/asr
lexicon_path=/home_new/SpeechT5/SpeechLM/dataset/LibriSpeech/hidden_unit/librispeech_lexicon.lst


# id_text=/home_new/SpeechT5/SpeechLM/dataset/LibriSpeech/hidden_unit/bin-idx/${file_name}_text.tsv
# t2u_manifest=/home_new/SpeechT5/SpeechLM/dataset/LibriSpeech/hidden_unit/bin-idx/${file_name}_phn
# output_phn=/home_new/SpeechT5/SpeechLM/dataset/LibriSpeech/hidden_unit/bin-idx/${file_name}.phn

set -e
mkdir -p dataset/LibriSpeech/fast_phone2unit/tmp && cd dataset/LibriSpeech/
# cd dataset/LibriSpeech/

echo "--------------------------------------------------------------------------------------"
echo "--------Manifest..."
echo "--------------------------------------------------------------------------------------"
python $src/wav2vec_manifest.py $dic_path --dest $dest --ext flac --valid-percent 0
mv $dest/train.tsv $dest/$file_name.tsv

echo "--------------------------------------------------------------------------------------"
echo "--------Labels..."
echo "--------------------------------------------------------------------------------------"
python $src/libri_labels.py $input_tsv --output-dir $dest --output-name $file_name

cd fast_phone2unit/

echo "--------------------------------------------------------------------------------------"
echo "--------Align the text order with tsv..."
echo "--------------------------------------------------------------------------------------"
python $src/align_text_order_with_tsv.py -i $input_text -t $input_tsv -o tmp/${file_name}_text.tsv

echo "--------------------------------------------------------------------------------------"
echo "--------Phoneize the text..."
echo "--------------------------------------------------------------------------------------"
python $src/phoneize_with_sil_librispeech.py -i tmp/${file_name}_text.tsv -o tmp/${file_name}.phn --lexicon $lexicon_path --surround -s 0.25

# cat ../librispeech-lm-norm.txt | sed '1d' | python $src/wrd2ltr.py > tmp/librilm.ltr

# echo "--------------------------------------------------------------------------------------"
# echo "--------Tokenize the text to the kaldi-style phonemes ..."
# echo "--------------------------------------------------------------------------------------"
# python $src/phoneme_tokenizer/ltr2kaldi_phn_sil025.py -i tmp/librilm.ltr -o tmp/librilm
# cat tmp/librilm.kaldi_phn_sil025 | sed 's/SIL_S/SIL/g' > tmp/librilm.phn

# echo "--------------------------------------------------------------------------------------"
# echo "--------Filter too long samples and up-sample phonemes ..."
# echo "--------------------------------------------------------------------------------------"
# python $src/filter_paireddata_by_len.py -i tmp/librilm -o tmp/librilm_l2k -s phn -t ltr -m 2000
# python $src/phoneme_tokenizer/repeat_withou_insert_sil_less_4375.py \
#     tmp/librilm_l2k.phn \
#     $src/phoneme_tokenizer/mean5_and_std25_sil14_spn32.dict \
#     tmp/librilm_l2k_upsample.phn

# mv tmp/librilm_l2k.ltr tmp/librilm_l2k_upsample.ltr 
# python $src/filter_paireddata_by_len.py -i tmp/librilm_l2k_upsample -o train_text.phn-ltr -s phn -t ltr -m 2800
### the max-length is set to filter the data, considering the batch size (in Large setting, 900,000/320 = 2812 tokens in a batch).

echo "--------------------------------------------------------------------------------------"
echo "--------Create namifest file ..."
echo "--------------------------------------------------------------------------------------"
python $src/get_t2u_manifest_textonly_Librispeech.py -i tmp/${file_name} -o ${file_name}_phn

cd $cwd
echo "--------------------------------------------------------------------------------------"
echo "--------Generate unit file ..."
echo "--------------------------------------------------------------------------------------"
gen_set=dataset/LibriSpeech/fast_phone2unit/${file_name}_phn
bash speechlm/scripts/tokenizer_fastT2U/generate.sh $gen_set

echo "--------------------------------------------------------------------------------------"
echo "--------Generate training file ..."
echo "--------------------------------------------------------------------------------------"
input_pseudo_label=/home_new/SpeechT5/SpeechLM/dataset/LibriSpeech/fast_phone2unit/pseudo_${file_name}_phn/0/generate-${file_name}_phn.txt
python $src/prepare_unitdata.py -i $input_pseudo_label -o $cwd/dataset/LibriSpeech/hidden_unit/${file_name}.km-ltr
cp $dest/${file_name}.ltr $cwd/dataset/LibriSpeech/hidden_unit/${file_name}.km-ltr.ltr

cd dataset/LibriSpeech/hidden_unit/
echo "--------------------------------------------------------------------------------------"
echo "--------Create binary files ..."
echo "--------------------------------------------------------------------------------------"
[ ! -f bin-idx/dict.km.txt ] && echo "dict ${cwd}/dataset/LibriSpeech/hidden_unit/bin-idx/dict.km.txt not found!" && exit 1
[ ! -f bin-idx/dict.ltr.txt ] && echo "dict ${cwd}/dataset/LibriSpeech/hidden_unit/bin-idx/dict.ltr.txt not found!" && exit 1
bash $src/txt2idx.sh ${file_name}.km-ltr.km bin-idx bin-idx/dict.km.txt
bash $src/txt2idx.sh ${file_name}.km-ltr.ltr bin-idx bin-idx/dict.ltr.txt

# rm -r tmp
cd -
echo "--------------------------------------------------------------------------------------"
echo "--------Done! files are in ${PWD}/dataset/LibriSpeech/hidden_unit/bin-idx"
echo "--------------------------------------------------------------------------------------"
