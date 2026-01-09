import torch

vocab_size = 8
output_dim = 6

torch.manual_seed(225)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)    #yaha pe 6 diffrent token ke liye  6x3 ka matrix banayega 

print(embedding_layer.weight)


print(embedding_layer(torch.tensor([6])))    #loopup 6 id 
print(embedding_layer(torch.tensor([1, 7, 3])))  #1,7,3 lookup

input_ids = torch.tensor([2, 3, 4, 1])
print(embedding_layer(input_ids))  #2,3,4,1 id to lookup