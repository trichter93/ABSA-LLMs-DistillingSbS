from google.colab import drive
drive.mount('/content/drive')
models_path = "/content/drive/MyDrive/ABSA-LLMs-DistillingSbS/models"
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    T5ForConditionalGeneration,
)

from model_utils import TaskPrefixDataCollator

def train_and_evaluate(run, tokenizer, tokenized_train, tokenized_valid, batch_size, grad_steps, base_model_name = 't5-large', training_method = 'distilling_sbs', lr =  1e-4, alpha = 0.5, epochs = 5, output_rationale = True, target_path = '/t5-DistillingSbS-ABSA'):
    full_path = models_path + target_path
    model = T5ForConditionalGeneration.from_pretrained(base_model_name)
    if training_method == "distilling_sbs":
        data_collator = TaskPrefixDataCollator(tokenizer=tokenizer, model=model)
    elif training_method == 'standard':
        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    else:
        raise ValueError
    set_seed(run)
    training_args = Seq2SeqTrainingArguments(
        output_dir = full_path,
        num_train_epochs=epochs,
        remove_unused_columns = False,
        evaluation_strategy = 'no',
        save_strategy='steps',
        save_steps = 5000,
        logging_dir = full_path,
        logging_strategy=logging_strategy,
        logging_steps=save_steps,
        learning_rate = lr,
        gradient_accumulation_steps=grad_steps,
        per_device_train_batch_size=batch_size,
        predict_with_generate=True,
        seed=run,
        local_rank=1,
        bf16=True,
        generation_max_length=512,
        prediction_loss_only=False,
      )
      
    trainer_kwargs = {
        'alpha': alpha,
        'output_rationale': output_rationale,
        'model': model,
        'args': training_args,
        'train_dataset': tokenized_train,
        'eval_dataset': {'test': tokenized_valid},
        'data_collator': data_collator,
        'tokenizer': tokenizer
    }


    if training_method == 'distilling_sbs':
        trainer = TaskPrefixTrainer(**trainer_kwargs)
    elif training_method == 'standard':
        trainer_kwargs.pop('alpha')
        trainer_kwargs.pop('output_rationale')
        trainer = Seq2SeqTrainer(**trainer_kwargs)
    else:
        raise ValueError


    return (trainer.train(), trainer)