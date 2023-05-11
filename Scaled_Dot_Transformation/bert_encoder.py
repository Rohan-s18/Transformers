"""
Author: Rohan Singh
Python Module to have fun with the encoder of Bert
"""

#  Imports
from transformers import BertModel


def encoder_layer(model, n):
    # retrieving the "n-th" encoder from the stack
    layer = model.encoder.layer[n]  
    print(layer)


#  Minn function for demonstration
def main():
    
    model = BertModel.from_pretrained('bert-base-uncased')
    
    print(len(model.encoder.layer))

    encoder_layer(model, 0)
    


if __name__ == "__main__":
    main()
