"""
Author: Rohan Singh
Python Module for tokenization using the huggingface API
"""


#  Import
from transformers import BertModel, BertTokenizer
import torch
from sklearn.metrics.pairwise import cosine_similarity 

"""
Functions for tokenization
"""


#  This fucntion checks if a set of word belongs to the training set for BERT-base
def check_in_vocab(words, tokenizer):
    for word in words:
        print(f"is {word} in the BERT-base vocabulary: {word in tokenizer.vocab}")


#  This function returns the token information for a given text
def token_info(text, tokenizer):
    tokens = tokenizer.encode(text)
    # Printing the tokens and the subwords
    print(f"Text: {text}\nNumber of Tokens in this text: {len(tokens)}\n")
    for token in tokens:
        print(f"Token: {token}, subword: {tokenizer.decode([token])}")


#  This function returns the cosine similarity between 2 words (wrt to context)
def get_similarity(model, tokenizer, text_1, text_2, pos_1, pos_2):

    #Finding the embedding for word 1
    w1_embedding = model(torch.tensor(tokenizer.encode(text_1)).unsqueeze(0))[0][:, pos_1, :].detach().numpy()

    #Finding the embedding for word 2
    w2_embedding = model(torch.tensor(tokenizer.encode(text_2)).unsqueeze(0))[0][:, pos_2, :].detach().numpy()

    #  Calculating the cosine similarity between the embeddings
    return cosine_similarity(w1_embedding, w2_embedding)



#  Main function for demonstration
def main():

    #  Creating the tokenizer object
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


    #  Creating the model object
    model = BertModel.from_pretrained('bert-base-uncased')


    #  Using vocab checker
    names = ["rachel","rohan","sid","zeeshan","ashwin"]
    check_in_vocab(names, tokenizer)
    print("\n")


    #  Getting the token info for this piece of text
    text = "rohan and rachel love doing transformer stuff together"
    token_info(text, tokenizer)
    print("\n")


    #  Getting the cosine similarity of two words in context
    t_1 = "rohan has a pet python"
    t_2 = "rohan likes coding in python"
    similarity = get_similarity(model, tokenizer, t_1, t_2, 5, 5)
    print(f"The similarity between the 2 instances of the word python in context is: {similarity}\n")
    




if __name__ == "__main__":
    main()


