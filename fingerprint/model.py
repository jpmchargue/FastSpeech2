import numpy as np
import torch
import torch.nn as nn


class Parameters():
    DATA_PATH = "data/speakers"
    
    verifier_hidden_size = 256
    verifier_utterances_per_speaker = 10


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
params = Parameters()

class Verifier(nn.Module):
    def __init__(self):
        super(Verifier, self).__init__()
        self.hidden_size = params.verifier_hidden_size
        self.lstm = nn.LSTM(40, self.hidden_size, 3, batch_first=True).to(device)

        self.similarity_weight = nn.Parameter(torch.tensor([10.])).to(device)
        self.similarity_bias = nn.Parameter(torch.tensor([-5.])).to(device)
        self.loss_function = nn.CrossEntropyLoss().to(device)

    def forward(self, input):
        output, (hidden, cell_state) = self.lstm(input)
        #linear = self.linear(hidden[-1])
        #encoded = self.relu(linear)
        #return encoded / (torch.norm(encoded, dim=1, keepdim=True) + 1e-5)
        return hidden[-1]
    
    def initHidden(self, num_batches):
        return torch.zeros(3, num_batches, self.hidden_size)

    def initCellState(self, num_batches):
        return torch.zeros(3, num_batches, self.hidden_size)

    # Snippet from https://github.com/CorentinJ/Real-Time-Voice-Cloning
    def similarity_matrix(self, embeds):
        """
        Computes the similarity matrix according the section 2.1 of GE2E.
        :param embeds: the embeddings as a tensor of shape (speakers_per_batch, 
        utterances_per_speaker, embedding_size)
        :return: the similarity matrix as a tensor of shape (speakers_per_batch,
        utterances_per_speaker, speakers_per_batch)
        """
        speakers_per_batch, utterances_per_speaker = embeds.shape[:2]
        
        # Inclusive centroids (1 per speaker). Cloning is needed for reverse differentiation
        centroids_incl = torch.mean(embeds, dim=1, keepdim=True)
        centroids_incl = centroids_incl.clone() / (torch.norm(centroids_incl, dim=2, keepdim=True) + 1e-5)

        # Exclusive centroids (1 per utterance)
        centroids_excl = (torch.sum(embeds, dim=1, keepdim=True) - embeds)
        centroids_excl /= (utterances_per_speaker - 1)
        centroids_excl = centroids_excl.clone() / (torch.norm(centroids_excl, dim=2, keepdim=True) + 1e-5)

        # Similarity matrix. The cosine similarity of already 2-normed vectors is simply the dot
        # product of these vectors (which is just an element-wise multiplication reduced by a sum).
        # We vectorize the computation for efficiency.
        sim_matrix = torch.zeros(speakers_per_batch, utterances_per_speaker,
                                 speakers_per_batch).to(device)
        mask_matrix = 1 - np.eye(speakers_per_batch, dtype=int)
        for j in range(speakers_per_batch):
            mask = np.where(mask_matrix[j])[0]
            sim_matrix[mask, :, j] = (embeds[mask] * centroids_incl[j]).sum(dim=2)
            sim_matrix[j, :, j] = (embeds[j] * centroids_excl[j]).sum(dim=1)
        
        ## Even more vectorized version (slower maybe because of transpose)
        # sim_matrix2 = torch.zeros(speakers_per_batch, speakers_per_batch, utterances_per_speaker
        #                           ).to(device)
        # eye = np.eye(speakers_per_batch, dtype=np.int)
        # mask = np.where(1 - eye)
        # sim_matrix2[mask] = (embeds[mask[0]] * centroids_incl[mask[1]]).sum(dim=2)
        # mask = np.where(eye)
        # sim_matrix2[mask] = (embeds * centroids_excl).sum(dim=2)
        # sim_matrix2 = sim_matrix2.transpose(1, 2)
        
        sim_matrix = sim_matrix * self.similarity_weight + self.similarity_bias
        return sim_matrix
    
    def new_similarity_matrix(self, embeds):
        centroids = torch.mean(embeds, 1)
        centroids = torch.nn.functional.normalize(centroids, dim=1)
        norms = torch.nn.functional.normalize(embeds, dim=2)

        num_speakers = 64
        num_utterances = 10
        sim_matrix = torch.zeros(num_speakers, num_utterances, num_speakers).to(device)
        for s in range(num_speakers):
            sim_matrix[:, :, s] = (norms * centroids[s]).sum(dim=2)

        sim_matrix = sim_matrix * self.similarity_weight + self.similarity_bias
        return sim_matrix


    def new_loss(self, embeds):
        num_speakers = 64
        num_utterances = 10
        sim_matrix = self.new_similarity_matrix(embeds)
        sim_matrix = sim_matrix.reshape((num_speakers * num_utterances, 
                                         num_speakers))
        ground_truth = np.repeat(np.arange(num_speakers), num_utterances)
        target = torch.from_numpy(ground_truth).long().to(device)

        loss = self.loss_function(sim_matrix, target)
        return loss


    def loss(self, embeds):
        """
        Computes the softmax loss according the section 2.1 of GE2E.
        
        :param embeds: the embeddings as a tensor of shape (speakers_per_batch, 
        utterances_per_speaker, embedding_size)
        :return: the loss and the EER for this batch of embeddings.
        """
        speakers_per_batch, utterances_per_speaker = embeds.shape[:2]
        
        # Loss
        sim_matrix = self.similarity_matrix(embeds)
        sim_matrix = sim_matrix.reshape((speakers_per_batch * utterances_per_speaker, 
                                         speakers_per_batch))
        ground_truth = np.repeat(np.arange(speakers_per_batch), utterances_per_speaker)
        target = torch.from_numpy(ground_truth).long().to(device)
        loss = self.loss_function(sim_matrix, target)

        return loss

    def get_embedding(self, spectrogram):
        """
        Compute the overall embedding vector for a sound sample,
        as inferred by the verifier network.
        This is calculated by finding the embedding vector for each point
        of a sliding window of length 160 with 50% overlap.
        Then, all embedding vectors are L2-normalized and averaged.
        """
        L = len(spectrogram)
        d_vector = torch.zeros(params.verifier_hidden_size).to(device)
        count = 0
        i = 0
        while (i + 140 < L):
            input = torch.from_numpy(spectrogram[i:i+140]).to(device)
            input = torch.unsqueeze(input, dim=0)
            embedding = self.forward(input)
            norm = embedding / torch.linalg.norm(embedding)
            d_vector = d_vector + norm
            count += 1
            i += 70
        d_vector = d_vector.view(-1)
        return d_vector.detach().cpu() / count

    def get_grad_embedding(self, spectrogram):
        """
        Compute the overall embedding vector for a sound sample,
        as inferred by the verifier network.
        This is calculated by finding the embedding vector for each point
        of a sliding window of length 160 with 50% overlap.
        Then, all embedding vectors are L2-normalized and averaged.
        """
        L = len(spectrogram)
        print(L)
        d_vector = torch.zeros(params.verifier_hidden_size).to(device)
        count = 0
        i = 0
        while (i + 140 < L):
            input = spectrogram[i:i+140].to(device)
            input = torch.unsqueeze(input, dim=0)
            embedding = self.forward(input)
            norm = embedding / torch.linalg.norm(embedding)
            d_vector = d_vector + norm
            count += 1
            i += 70
        d_vector = d_vector.view(-1)
        return d_vector / count