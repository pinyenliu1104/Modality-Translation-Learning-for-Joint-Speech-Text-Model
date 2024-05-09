#####################################
# Fast Text2Unit Model #
#####################################
# [ $# -lt 2 ] && echo "Usage: $0 <model_path> <gen_set> [outdir={gen_set%/*}]" && exit 0
[ ${PWD##*/} != SpeechLM ] && echo "Error: dir not match! Switch to SpeechLM/ and run it again!" && exit 1

# model_path=$1
model_path=/home_new/SpeechT5/SpeechLM/models/fastT2U_small_4gpu_ls0.1_tristage_lr5e-4_checkpoint_best.pt
src_dir=${model_path%/*}
cpt=${model_path##*/}
cpt=${cpt%.*}

gen_set=$1
# gen_set=/home_new/SpeechT5/SpeechLM/dataset/LibriSpeech/fast_phone2unit/train_clean_100_phn
outdir=$2

DATA_DIR=${gen_set%/*}
gen_set=${gen_set##*/}
[ -z $outdir ] && outdir=${DATA_DIR}

CODE_ROOT=${PWD}

nj=1
for rank in $(seq 0 $((nj-1))); do
    results_path=$outdir/pseudo_${gen_set}/${rank}
    [ ! -d $results_path ] && mkdir -p $results_path
    echo "$model_path" > $results_path/model.record

    python $CODE_ROOT/speechlm/generate_unit.py $DATA_DIR \
    --user-dir $CODE_ROOT/speechlm \
    --config-yaml config_generate.yaml \
    --path ${model_path} \
    --task fast_text_to_unit \
    --gen-subset $gen_set \
    \
    --beam 1 \
    --max-tokens 10000 \
    --results-path $results_path \
    --scoring sacrebleu \
    --skip-invalid-size-inputs-valid-test \
    --distributed-world-size $nj --distributed-rank ${rank} \
    &
done
wait
