import numpy as np

class LSTM:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Initialize weights and biases
        self.Wf = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wi = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wc = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wo = np.random.randn(hidden_size, input_size + hidden_size)

        self.bf = np.zeros((hidden_size, 1))
        self.bi = np.zeros((hidden_size, 1))
        self.bc = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))

    def forward(self, x, initial_hidden_state, initial_cell_state):
        """
        Processes a sequence of inputs and returns:
        - all hidden states
        - final hidden state
        - final cell state
        """

        h = initial_hidden_state
        c = initial_cell_state

        hidden_states = []

        def sigmoid(z):
            return 1 / (1 + np.exp(-z))

        for x_t in x:

            # Reshape current input to column vector
            x_t = x_t.reshape(-1, 1)

            # Combine previous hidden state and current input
            combined = np.vstack((h, x_t))

            # Forget gate
            f = sigmoid(
                self.Wf @ combined + self.bf
            )

            # Input gate
            i = sigmoid(
                self.Wi @ combined + self.bi
            )

            # Candidate cell state
            c_candidate = np.tanh(
                self.Wc @ combined + self.bc
            )

            # Output gate
            o = sigmoid(
                self.Wo @ combined + self.bo
            )

            # Update cell state
            c = f * c + i * c_candidate

            # Update hidden state
            h = o * np.tanh(c)

            # Store hidden state
            hidden_states.append(h.copy())

        return np.array(hidden_states), h, c