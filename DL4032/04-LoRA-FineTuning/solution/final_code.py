import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from datasets import load_dataset
import evaluate

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

# Load model
config = AutoConfig.from_pretrained('gpt2')
model = AutoModelForCausalLM.from_pretrained("gpt2", config=config)

# Load dataset
dataset = load_dataset("squad")
vali_ds = dataset['train'].select(range(5))
split_ds = dataset['train'].train_test_split(test_size=0.1)
train_ds = split_ds['train'].shuffle(seed=42).select(range(2000))
eval_ds = split_ds['test'].shuffle(seed=42).select(range(200))

# Clear memory
del dataset, split_ds
gc.collect()

# FIXED: Simplified preprocessing function
def preprocess_function(examples):
    """
    Creates properly aligned input-label pairs for causal LM fine-tuning
    """
    inputs = []
    targets = []
    
    for context, question, answers in zip(examples['context'], examples['question'], examples['answers']):
        # Get the answer text (use first answer if multiple)
        answer_text = answers['text'][0] if answers['text'] else ""
        
        # Create full sequence: prompt + answer
        prompt = f"Context: {context}\nQuestion: {question}\nAnswer: "
        full_text = prompt + answer_text + tokenizer.eos_token
        
        inputs.append(full_text)
    
    # Tokenize all at once
    model_inputs = tokenizer(
        inputs,
        truncation=True,
        max_length=256,
        padding="max_length",
        return_tensors=None
    )
    
    # Create labels by copying input_ids
    labels = []
    for i, input_ids in enumerate(model_inputs['input_ids']):
        # Create labels
        label_ids = input_ids.copy()
        
        # Find answer start position for masking
        prompt = f"Context: {examples['context'][i]}\nQuestion: {examples['question'][i]}\nAnswer: "
        prompt_ids = tokenizer(prompt, add_special_tokens=False)['input_ids']
        prompt_length = len(prompt_ids)
        
        # Mask prompt tokens with -100
        for j in range(min(prompt_length, len(label_ids))):
            label_ids[j] = -100
        
        labels.append(label_ids)
    
    model_inputs['labels'] = labels
    return model_inputs

# FIXED: Custom data collator for proper padding
from transformers import DataCollatorForLanguageModeling

class QADataCollator:
    def __init__(self, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __call__(self, features):
        # Pad sequences
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        
        for feature in features:
            input_ids = feature['input_ids']
            attention_mask = feature['attention_mask'] 
            labels = feature['labels']
            
            # Pad to max_length
            padding_length = self.max_length - len(input_ids)
            
            if padding_length > 0:
                input_ids.extend([self.tokenizer.pad_token_id] * padding_length)
                attention_mask.extend([0] * padding_length)
                labels.extend([-100] * padding_length)  # Ignore padded tokens in loss
            else:
                # Truncate if too long
                input_ids = input_ids[:self.max_length]
                attention_mask = attention_mask[:self.max_length] 
                labels = labels[:self.max_length]
            
            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels)
        
        return {
            'input_ids': torch.tensor(batch_input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(batch_attention_mask, dtype=torch.long),
            'labels': torch.tensor(batch_labels, dtype=torch.long)
        }

# Apply preprocessing
tok_train_ds = train_ds.map(preprocess_function, batched=True, remove_columns=train_ds.column_names)
tok_eval_ds = eval_ds.map(preprocess_function, batched=True, remove_columns=eval_ds.column_names)

print(f"Training examples: {len(tok_train_ds)}")
print(f"Eval examples: {len(tok_eval_ds)}")

# Setup LoRA with proper gradient setup
from peft import LoraConfig, TaskType, get_peft_model

# Ensure base model parameters are frozen
for param in model.parameters():
    param.requires_grad = False

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    fan_in_fan_out=False,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["c_attn"]
)

lora_model = get_peft_model(model, lora_config)
lora_model.print_trainable_parameters()

# Verify gradients are enabled for LoRA parameters
for name, param in lora_model.named_parameters():
    if param.requires_grad:
        print(f"Trainable: {name}")

lora_model.train()

# Create data collator
data_collator = QADataCollator(tokenizer, max_length=256)

# Training setup
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    output_dir="./results",
    learning_rate=3e-4,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    num_train_epochs=10,
    logging_steps=50,
    warmup_steps=100,
    gradient_checkpointing=False,  # Disable gradient checkpointing
    dataloader_pin_memory=False,
    remove_unused_columns=False,
    report_to="none"
)

trainer = Trainer(
    model=lora_model,
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=tok_train_ds,
    eval_dataset=tok_eval_ds
)

# Train
print("Starting training...")
trainer.train()

# Evaluate
evaluation_result = trainer.evaluate()
print("Evaluation results:", evaluation_result)

# Save model
lora_model.save_pretrained("gpt2-lora-fixed")

# FIXED: Proper inference function
def generate_answer(model, tokenizer, context, question, max_new_tokens=50):
    """Generate answer for a given context and question"""
    prompt = f"Context: {context}\nQuestion: {question}\nAnswer: "
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=200)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode only the generated part (after the prompt)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = generated_text[len(prompt):].strip()
    
    return answer

# Load and test fine-tuned model
from peft import AutoPeftModelForCausalLM
ft_model = AutoPeftModelForCausalLM.from_pretrained("gpt2-lora-fixed")

# Test on validation examples
print("\n=== Testing Fine-tuned Model ===")
for i, example in enumerate(vali_ds):
    if i >= 3:  # Test first 3 examples
        break
        
    context = example['context']
    question = example['question']
    true_answers = example['answers']['text']
    
    # Generate answer
    predicted_answer = generate_answer(ft_model, tokenizer, context, question)
    
    print(f"\n--- Example {i+1} ---")
    print(f"Question: {question}")
    print(f"True Answer: {true_answers[0] if true_answers else 'N/A'}")
    print(f"Predicted: {predicted_answer}")
    print("-" * 50)

# Formal evaluation with SQuAD metrics
predictions = []
references = []

for example in vali_ds:
    predicted_answer = generate_answer(ft_model, tokenizer, example['context'], example['question'])
    
    predictions.append({
        'id': example['id'],
        'prediction_text': predicted_answer
    })
    
    references.append({
        'id': example['id'],
        'answers': example['answers']
    })

# Load and compute SQuAD metrics
try:
    squad_metric = evaluate.load("squad")
    results = squad_metric.compute(predictions=predictions, references=references)
    print(f"\nSQuAD Metrics: {results}")
except Exception as e:
    print(f"Could not load SQuAD metric: {e}")
    print("Install with: pip install datasets[metrics]")
