import argparse
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, TrainingArguments, EarlyStoppingCallback
from sklearn.model_selection import train_test_split
from datasets import load_metric
import json
import pdb
from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq
from process import parse_txt, write_new_file, as_json_format
import os

# Custom data collator
class CustomDataCollator():
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def collate_batch(self, batch):
        # Extract inputs and labels from the batch
        inputs = ["generate ASR error : "+ item['gold'] for item in batch]
        labels = [item['Pred'] for item in batch]

        # Tokenize inputs and labels
        inputs = self.tokenizer(inputs, return_tensors='pt', padding=True, truncation=True)
        labels = self.tokenizer(labels, return_tensors='pt', padding=True, truncation=True)

        collated_batch = {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'labels': labels['input_ids'],
        }

        return collated_batch
    
    
def main():
    parser = argparse.ArgumentParser(description="Train T5-based correction model.")
    parser.add_argument("--output_dir", type=str, default="./t5_correction_model", help="Output directory for the trained model.")
    parser.add_argument("--num_train_epochs", type=float, default=3, help="Number of training epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="Batch size per GPU/CPU during training.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4, help="Batch size per GPU/CPU during evaluation.")
    parser.add_argument("--save_steps", type=int, default=1000, help="Save model checkpoint every X steps.")
    parser.add_argument("--eval_steps", type=int, default=500, help="Evaluate model every X steps.")
    parser.add_argument("--logging_dir", type=str, default="./logs", help="Directory for logging.")
    parser.add_argument("--data_path", type =str)
    parser.add_argument("--predict_path", type =str)
    parser.add_argument("--model_name_or_path", type =str)
    args = parser.parse_args()

    dataset = json.load(open(args.data_path, "r"))
    train_dataset, val_dataset = train_test_split(dataset, test_size=0.1, random_state=42)
    predict_dial = parse_txt(args.predict_path)
    predict_dataset = as_json_format(predict_dial)
    # Load T5 tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)

    # Define training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        save_steps=args.save_steps,
        save_total_limit=1,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        logging_dir=args.logging_dir,
        logging_steps=100,
        remove_unused_columns=False,
        load_best_model_at_end=True,
        predict_with_generate=True,
    )


    # Initialize Trainer
    # Add early stopping callback
    early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=3)

    custom_data_collator = CustomDataCollator(tokenizer).collate_batch
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=custom_data_collator,
        callbacks=[early_stopping_callback], # early stopping
        
    )

    # Train the model
    trainer.train()

    ### SAVE ###
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    ### PREDICT ###
    metric = load_metric("cer")
    predictions = trainer.predict(predict_dataset)
    predictions = tokenizer.batch_decode(predictions.predictions, skip_special_tokens=True)
    references = [item['gold'] for item in predict_dataset]
    save_file_path = os.path.join(args.output_dir, "predictions.txt")
    write_new_file(predictions, predict_dial, save_file_path)
    
    
    ### METRIC ###
    cer = metric.compute(predictions=predictions, references=references)

    metric = {'cer': cer}   
    json.dump(metric, open(os.path.join(args.output_dir, "metrics.json"), "w"))
    print("save all in ", args.output_dir)
if __name__ == "__main__":
    main()
