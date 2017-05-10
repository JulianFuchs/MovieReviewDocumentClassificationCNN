# MovieReviewDocumentClassificationCNN
Uses a word based CNN to classify Movie Reviews

The Movie Review dataset (polarity dataset v2.0) from http://www.cs.cornell.edu/people/pabo/movie-review-data/ is used, which provides 1000 positive and 1000 negative reviews.

The CNN and methodology introduced in 'Neural Document Embeddings for Intensive Care Patient Mortality Prediction' by Paulina Grnarova, Florian Schmidt, Stephanie L. Hyland and Carsten Eickhoff (https://arxiv.org/pdf/1612.00467.pdf is) is used to classify the reviews.

It uses 2 seperate CNNs. A first sentence level CNN takes seperate sentences from a review, applies convolution and pooling and returns a sentence representation. All sentence represenations from a review are combined and a document CNN is applied, resulting in the final classification. 
