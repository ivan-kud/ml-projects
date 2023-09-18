### Objective
Develop a tool that analyses a given text and suggests improvements based on the similarity to a list of "standardised" phrases. These standardised phrases represent the ideal way certain concepts should be articulated, and the tool should recommend changes to align the input text closer to these standards.

### Design decision
Decision is based on the pretrained RoBERTa model from Hugging Face library.  Embedding vectors were calculated as a mean polling value of RoBERTa’s last hidden state.

### Algorithm description
Embedding vectors were computed for each of "standardised" phrases as a mean polling value of RoBERTa’s last hidden state.

The given text was also fed to RoBERTa model and the last hidden state was used to calculate different context embeddings as a mean polling value.

Similarity sore was calculated as cosine distance for each embedding pair (standard phrase and text context).

### Results
A simple [UI web-service](https://ivankud.com/suggestion) was deployed to test designed tool.