import torch
from transformer import MiniTransformer

# Cargar checkpoint
checkpoint = torch.load("mini_transformer.pt")

# Reconstruir vocabulario
vocab = checkpoint['vocab']
idx_to_word = {i: w for w, i in vocab.items()}

# Crear modelo con parámetros idénticos
vocab_size = len(vocab)
model = MiniTransformer(vocab_size, d_model=32, num_heads=4, d_ff=128, num_layers=2)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()  # poner en modo evaluación

print("Modelo cargado correctamente.")

# Generar texto
generated = torch.tensor([[vocab["el"]]])
for _ in range(8):
    logits = model(generated)
    probs = torch.softmax(logits[:, -1, :], dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    generated = torch.cat([generated, next_token], dim=1)

print("Frase generada:")
print(' '.join(idx_to_word[i.item()] for i in generated[0]))
