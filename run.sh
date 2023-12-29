SCRIPT_PATH="./run.py"

# Set the path to your data file
DATA_PATH="/home/jihyunlee/ASR/weak_asr_datas/result_cm3/10800/1/asr_results.jsonl"
PREDICT_PATH="/home/jihyunlee/ASR/asr/manifests/train_manifests/augmented/train_manifest.txt"
# PREDICT_PATH="/home/jihyunlee/ASR/asr/manifests/train_manifests/small.txt"

MODEL="t5-base"
# Run the Python script with the specified arguments
python $SCRIPT_PATH --output_dir=./exps/$MODEL-tune \
                   --num_train_epochs=100 \
                   --per_device_train_batch_size=16 \
                   --per_device_eval_batch_size=8 \
                   --save_steps=500 \
                   --eval_steps=500 \
                   --logging_dir="./logs" \
                   --data_path=$DATA_PATH \
                   --predict_path=$PREDICT_PATH \
                   --model_name_or_path=$MODEL \
