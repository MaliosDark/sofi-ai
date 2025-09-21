from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('./SOFIA')
print('SOFIA Chat Simulator (Retrieval mode)')
print('Type your message, or "quit" to exit.')
print('The model will "respond" with the most similar example sentence.')

# Some example sentences for similarity (acting as responses)
examples = [
    "Hello! How can I help you today?",
    "The weather is nice, isn't it?",
    "Why did the chicken cross the road? To get to the other side!",
    "This is an embedding model that finds similar texts.",
    "Goodbye! Have a great day."
]
example_embs = model.encode(examples)

while True:
    try:
        user_input = input('You: ')
        if user_input.lower() == 'quit':
            break
        emb = model.encode([user_input])

        # Compute cosine similarities
        emb_norm = emb / np.linalg.norm(emb)
        example_embs_norm = example_embs / np.linalg.norm(example_embs, axis=1, keepdims=True)
        similarities = np.dot(emb_norm, example_embs_norm.T)[0]
        best_idx = np.argmax(similarities)
        response = examples[best_idx]
        print(f'SOFIA: {response}')
        print(f'(Similarity: {similarities[best_idx]:.3f})')
        print()
    except KeyboardInterrupt:
        break
