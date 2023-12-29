# Set the path to your Python script
SCRIPT_PATH="./run.py"

# Set the path to your data file
DATA_PATH="/home/jihyunlee/ASR/weak_asr_datas/result_cm3/10800/1/asr_results.jsonl"

# Run the Python script with the specified arguments
python $SCRIPT_PATH --output_dir="./t5_correction_model" \
                   --num_train_epochs=3 \
                   --per_device_train_batch_size=4 \
                   --per_device_eval_batch_size=4 \
                   --save_steps=1000 \
                   --eval_steps=500 \
                   --logging_dir="./logs" \
                   --data_path=$DATA_PATH
