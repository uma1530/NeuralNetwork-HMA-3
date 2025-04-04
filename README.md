# NeuralNetwork-HMA-3

Q1: Implementing a Basic Autoencoder
Objective:
Autoencoders learn to encode input data into a lower-dimensional representation and reconstruct it back to its original form. This question focuses on building a simple autoencoder and analyzing how different latent space dimensions affect reconstruction quality.
Steps:
1.	Load the MNIST dataset
o	Use tensorflow.keras.datasets.mnist.load_data()
o	Normalize pixel values to range [0,1].
o	Flatten images from (28,28) to (784,) for input to a dense neural network.
2.	Define the Autoencoder Architecture
o	Encoder: Input layer (784 neurons) → Dense layer (32 neurons) → Latent representation (32).
o	Decoder: Dense layer (32 neurons) → Output layer (784 neurons) reshaped to (28,28).
o	Use Model() from tensorflow.keras.models to define both encoder and decoder.
3.	Compile and Train the Autoencoder
o	Loss: binary_crossentropy
o	Optimizer: adam
o	Use model.fit() on the MNIST dataset.
4.	Visualize Reconstruction Performance
o	Compare original vs. reconstructed images by plotting them.
5.	Experiment with Different Latent Dimensions (16, 64, etc.)
o	Observe how decreasing or increasing latent space size affects reconstruction quality.
Key Insights:
•	Smaller latent space (e.g., 16) may result in lossy reconstruction.
•	Larger latent space (e.g., 64) may retain more image details.

Q2: Implementing a Denoising Autoencoder
Objective:
A denoising autoencoder learns to reconstruct clean images from noisy inputs, making it useful in noise removal applications like medical imaging and document restoration.
Steps:
1.	Modify the Basic Autoencoder
o	Keep the same architecture but introduce noisy images as inputs while keeping the original images as targets.
2.	Add Gaussian Noise to Input Images
o	Use np.random.normal(loc=0, scale=0.5, size=image.shape) to add noise.
o	Clip values to [0,1] to maintain valid pixel range.
3.	Train the Model
o	Use binary_crossentropy loss.
o	Train the model so it learns to remove noise while reconstructing the images.
4.	Compare Noisy vs. Reconstructed Images
o	Plot noisy images, reconstructed images, and clean images for visual comparison.
5.	Compare with Basic Autoencoder
o	Check if the denoising autoencoder performs better than a standard autoencoder when noise is present.
6.	Real-World Application
o	Medical Imaging: Enhancing MRI or X-ray scans by removing noise.
o	Security: Restoring blurred or distorted surveillance images.
Key Insights:
•	Denoising autoencoders can generalize better and perform better on real-world noisy data.



Q3: Implementing an RNN for Text Generation
Objective:
Recurrent Neural Networks (RNNs) like LSTMs can learn patterns in text sequences and generate new text character-by-character or word-by-word.
Steps:
1.	Load a Text Dataset
o	Use a dataset like Shakespeare's Sonnets or "The Little Prince".
o	Read the text file and preprocess it (lowercasing, removing special characters if necessary).
2.	Convert Text into Sequences
o	Tokenize text into characters or words.
o	Use one-hot encoding or embeddings (Embedding() layer in Keras).
o	Create input-output pairs for training (e.g., given "hell", predict "o").
3.	Define an LSTM-based RNN
o	Model structure:
	Embedding() layer for character/word representation.
 LSTM() layers for sequence modeling.
	Dense() layer with softmax activation for next-character prediction.
4.	Train the Model
o	Loss: categorical_crossentropy
o	Optimizer: adam
o	Use model.fit() on text sequences.
5.	Generate Text
o	Start with a seed sequence.
o	Predict next characters iteratively and append to the generated text.
6.	Explain Temperature Scaling in Text Generation
o	Low Temperature (e.g., 0.2): Less randomness, model selects high-probability words → repetitive and predictable output.
o	High Temperature (e.g., 1.2): More randomness, model explores diverse outputs but may generate gibberish.
Key Insights:
•	Higher LSTM units allow better learning but may increase overfitting.
•	Using embeddings instead of one-hot encoding improves efficiency.




Q4: Sentiment Classification Using RNN
Objective:
Train an LSTM-based sentiment classifier to determine if a movie review is positive or negative using the IMDB dataset.
Steps:
1.	Load the IMDB Dataset
o	Use tensorflow.keras.datasets.imdb.load_data()
o	The dataset consists of 50,000 movie reviews labeled as positive or negative.
2.	Preprocess Text Data
o	Tokenize reviews into integer sequences (Tokenizer() from Keras).
o	Pad sequences to ensure uniform input length (pad_sequences()).
3.	Define an LSTM-Based Model
o	Embedding() layer to convert words into dense vectors.
o	LSTM() layer to capture sequence patterns.
o	Dense() layer with sigmoid activation for binary classification.
4.	Train the Model
o	Loss: binary_crossentropy
o	Optimizer: adam
o	Use model.fit() to train on IMDB data.
5.	Evaluate the Model with a Confusion Matrix & Classification Report
o	Compute confusion matrix using confusion_matrix() from sklearn.metrics.
o	Generate a classification report (accuracy, precision, recall, F1-score).
6.	Explain the Precision-Recall Tradeoff
o	Precision: Percentage of predicted positives that are actual positives.
o	Recall: Percentage of actual positives that are correctly predicted.
o	Tradeoff:
	If high precision is prioritized, recall may suffer.
	If high recall is prioritized, more false positives may occur.
o	Important for applications like fraud detection (high recall needed) or spam detection (high precision needed).
Key Insights:
•	LSTM-based sentiment classifiers can outperform traditional methods like bag-of-words.
•	Imbalanced datasets may require additional techniques like class weighting.
 
Final Takeaways:
1.	Autoencoders compress and reconstruct data; adding noise enables denoising capabilities.
2.	RNNs (LSTMs) can model sequences, allowing text generation and sentiment classification.
3.	Performance Metrics (e.g., loss, accuracy, precision-recall tradeoff) are crucial for evaluating models.
4.	Hyperparameter Tuning (e.g., latent space size, LSTM units, temperature scaling) impacts model performance.
Implementing an RNN for Text Generation

Task: Recurrent Neural Networks (RNNs) can generate sequences of text. You will train an LSTM-based RNN to predict the next character in a given text dataset.

1.	Load a text dataset (e.g., "Shakespeare Sonnets", "The Little Prince").
2.	Convert text into a sequence of characters (one-hot encoding or embeddings).
3.	Define an RNN model using LSTM layers to predict the next character.
4.	Train the model and generate new text by sampling characters one at a time.
5.	Explain the role of temperature scaling in text generation and its effect on randomness.








