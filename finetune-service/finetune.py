# finetune-service/finetune.py
import pandas as pd
from sentence_transformers import SentenceTransformer, models, InputExample, losses
from torch.utils.data import DataLoader

# Load your CSV
df = pd.read_csv("/app/data/products.csv")

# Create training examples
train_examples = []
for _, row in df.iterrows():
    text = f"{row['title']} {row['description']}"
    train_examples.append(InputExample(texts=[text, text], label=1.0))

# Setup model for fine-tuning
model_name = "all-MiniLM-L6-v2"
word_embedding_model = models.Transformer(model_name)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# Prepare DataLoader
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

# Loss function
train_loss = losses.CosineSimilarityLoss(model=model)

# Fine-tune the model
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=1,
    warmup_steps=10,
    output_path="/models/fine-tuned"
)
