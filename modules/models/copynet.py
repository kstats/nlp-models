import logging
from typing import Dict, Tuple

from overrides import overrides
import torch
from torch.nn.modules.linear import Linear
from torch.nn.modules.rnn import LSTMCell

from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Attention, TextFieldEmbedder, Seq2SeqEncoder
from allennlp.modules.token_embedders import Embedding
from allennlp.nn import util


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("copynet")
class CopyNet(Model):
    """
    This is an implementation of `CopyNet <https://arxiv.org/pdf/1603.06393>`_.
    CopyNet is a sequence-to-sequence encoder-decoder model with a copying mechanism
    that can copy tokens from the source sentence into the target sentence instead of
    generating all target tokens only from the target vocabulary.

    It is very similar to a typical seq2seq model used in neural machine translation
    tasks, for example, except that in addition to providing a "generation" score at each timestep
    for the tokens in the target vocabulary, it also provides a "copy" score for each
    token that appears in the source sentence. In other words, you can think of CopyNet
    as a seq2seq model with a dynamic target vocabulary that changes based on the tokens
    in the source sentence, allowing it to predict tokens that are out-of-vocabulary (OOV)
    with respect to the actual target vocab.

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        Vocabulary containing source and target vocabularies.
    source_embedder : ``TextFieldEmbedder``, required
        Embedder for source side sequences
    encoder : ``Seq2SeqEncoder``, required
        The encoder of the "encoder/decoder" model
    attention : ``Attention``, required
        This is used to get a dynamic summary of encoder outputs at each timestep
        when producing the "generation" scores for the target vocab.
    target_embedding_dim : ``int``, optional (default = 30)
        The size of the embeddings for the target vocabulary.
    copy_token : ``str``, optional (default = '@COPY@')
        The token used to indicate that a target token was copied from the source.
        If this token is not already in your target vocabulary, it will be added.
    source_namespace : ``str``, optional (default = 'source_tokens')
        The namespace for the source vocabulary.
    target_namespace : ``str``, optional (default = 'target_tokens')
        The namespace for the target vocabulary.
    """

    def __init__(self,
                 vocab: Vocabulary,
                 source_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 attention: Attention,
                 target_embedding_dim: int = 30,
                 copy_token: str = "@COPY@",
                 source_namespace: str = "source_tokens",
                 target_namespace: str = "target_tokens") -> None:
        super(CopyNet, self).__init__(vocab)
        self._source_namespace = source_namespace
        self._target_namespace = target_namespace
        self._start_index = self.vocab.get_token_index(START_SYMBOL, self._target_namespace)
        self._end_index = self.vocab.get_token_index(END_SYMBOL, self._target_namespace)
        self._copy_index = self.vocab.add_token_to_namespace(copy_token, self._target_namespace)
        self._oov_index = self.vocab.get_token_index(self.vocab._oov_token, self._target_namespace)  # pylint: disable=protected-access
        self._src_start_index = self.vocab.get_token_index(START_SYMBOL, self._source_namespace)
        self._src_end_index = self.vocab.get_token_index(END_SYMBOL, self._source_namespace)

        # Encoding modules.
        self._source_embedder = source_embedder
        self._encoder = encoder

        # Decoder output dim needs to be the same as the encoder output dim since we initialize the
        # hidden state of the decoder with the final hidden state of the encoder.
        # We arbitrarily set the decoder's input dimension to be the same as the output dimension.
        self.encoder_output_dim = self._encoder.get_output_dim()
        self.decoder_output_dim = self.encoder_output_dim
        self.decoder_input_dim = self.decoder_output_dim

        target_vocab_size = self.vocab.get_vocab_size(self._target_namespace)

        # The decoder input will be a function of the previous decoder output
        # (i.e. the hidden state from the last timestep), the embedding of the previous
        # predicted token, an attended encoder hidden state (called "attentive read"),
        # and another weighted sum of the encoder hidden state called the "selective read".
        self._target_embedder = Embedding(target_vocab_size, target_embedding_dim)
        self._attention = attention
        self._input_projection_layer = Linear(
                self.decoder_output_dim + target_embedding_dim + self.encoder_output_dim * 2,
                self.decoder_input_dim)

        # We then run the projected decoder input through an LSTM cell to produce
        # the next hidden state.
        self._decoder_cell = LSTMCell(self.decoder_input_dim, self.decoder_output_dim)

        # We create a "generation" score for each token in the target vocab
        # with a linear projection of the decoder hidden state.
        self._output_generation_layer = Linear(self.decoder_output_dim, target_vocab_size)

        # We create a "copying" score for each source token by applying a non-linearity
        # (tanh) to a linear projection of the encoded hidden state for that token,
        # and then taking the dot product of the result with the decoder hidden state.
        self._output_copying_layer = Linear(self.encoder_output_dim, self.decoder_output_dim)

    @overrides
    def forward(self,  # type: ignore
                source_tokens: Dict[str, torch.LongTensor],
                target_tokens: Dict[str, torch.LongTensor] = None,
                source_indices: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Make foward pass with decoder logic for producing the entire target sequence.

        Parameters
        ----------
        source_tokens : ``Dict[str, torch.LongTensor]``
            The output of `TextField.as_array()` applied on the source `TextField`. This will be
            passed through a `TextFieldEmbedder` and then through an encoder.
        target_tokens : ``Dict[str, torch.LongTensor]``, optional (default = None)
            Output of `Textfield.as_array()` applied on target `TextField`. We assume that the
            target tokens are also represented as a `TextField`.
        source_indices : ``Dict[str, torch.LongTensor]``, optional (default = None)
            Holds the index of every matching source token, for each token in the target
            sentence.

        Returns
        -------
        Dict[str, torch.Tensor]
        """
        state = self._init_encoded_state(source_tokens)

        if target_tokens:
            return self._forward_train(target_tokens, source_indices, state)

        return self._forward_predict(state)

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def _init_encoded_state(self, source_tokens: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Initialize the encoded state to be passed to the first decoding time step.
        """
        embedded_input = self._source_embedder(source_tokens)
        # shape: (batch_size, max_input_sequence_length, encoder_input_dim)

        batch_size, _, _ = embedded_input.size()

        source_mask = util.get_text_field_mask(source_tokens)
        # shape: (batch_size, max_input_sequence_length)

        encoder_outputs = self._encoder(embedded_input, source_mask)
        # shape: (batch_size, max_input_sequence_length, encoder_output_dim)

        final_encoder_output = util.get_final_encoder_states(
                encoder_outputs,
                source_mask,
                self._encoder.is_bidirectional())
        # shape: (batch_size, encoder_output_dim)

        # Initialize the decoder hidden state with the final output of the encoder.
        decoder_hidden = final_encoder_output
        # shape: (batch_size, decoder_output_dim)

        decoder_context = encoder_outputs.new_zeros(batch_size, self.decoder_output_dim)
        # shape: (batch_size, decoder_output_dim)

        state = {
                "source_mask": source_mask,
                "encoder_outputs": encoder_outputs,
                "decoder_hidden": decoder_hidden,
                "decoder_context": decoder_context
        }

        return state

    @staticmethod
    def _get_selective_weights(state: Dict[str, torch.Tensor]) -> torch.Tensor:
        group_size, source_sent_length, _ = state["encoder_outputs"].size()
        return state["encoder_outputs"].new_zeros((group_size, source_sent_length))

    def _decoder_step(self,
                      last_predictions: torch.Tensor,
                      selective_weights: torch.Tensor,
                      state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        encoder_outputs_mask = state["source_mask"].float()
        # shape: (batch_size, max_input_sequence_length, encoder_output_dim)

        embedded_input = self._target_embedder(last_predictions)
        # shape: (group_size, target_embedding_dim)

        attentive_weights = self._attention(
                state["decoder_hidden"], state["encoder_outputs"], encoder_outputs_mask)
        # shape: (batch_size, max_input_sequence_length)

        attentive_read = util.weighted_sum(state["encoder_outputs"], attentive_weights)
        # shape: (batch_size, encoder_output_dim)

        selective_read = util.weighted_sum(state["encoder_outputs"][:, 1:-1], selective_weights)
        # shape: (batch_size, encoder_output_dim)

        decoder_input = torch.cat((state["decoder_hidden"], embedded_input, attentive_read, selective_read), -1)
        # shape: (group_size, decoder_output_dim + target_embedding_dim + encoder_output_dim * 2)

        projected_decoder_input = self._input_projection_layer(decoder_input)
        # shape: (group_size, decoder_input_dim)

        state["decoder_hidden"], state["decoder_context"] = self._decoder_cell(
                projected_decoder_input,
                (state["decoder_hidden"], state["decoder_context"]))

        return state

    def _get_generation_scores(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self._output_generation_layer(state["decoder_hidden"])

    def _get_copy_scores(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        trimmed_encoder_outputs = state["encoder_outputs"][:, 1:-1]
        # (batch_size, max_input_sequence_length - 2, encoder_output_dim)

        copy_projection = self._output_copying_layer(trimmed_encoder_outputs)
        # (batch_size, max_input_sequence_length - 2, decoder_output_dim)

        copy_projection = torch.tanh(copy_projection)
        # (batch_size, max_input_sequence_length - 2, decoder_output_dim)

        copy_scores = copy_projection.bmm(state["decoder_hidden"].unsqueeze(-1)).squeeze(-1)
        # (batch_size, max_input_sequence_length - 2)

        return copy_scores

    def _get_ll_contrib(self,
                        generation_scores: torch.Tensor,
                        copy_scores: torch.Tensor,
                        target_tokens: torch.Tensor,
                        source_indices: torch.Tensor,
                        copy_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Numerically stable way of getting the log-likelihood contribution from a single timestep.

        Parameters
        ----------
        generation_scores : ``torch.Tensor``, (batch_size, target_vocab_size)
        copy_scores : ``torch.Tensor``, (batch_size, trimmed_source_length)
        target_tokens : ``torch.Tensor``, (batch_size,)
        source_indices : ``torch.Tensor``, (batch_size, trimmed_source_length)
        copy_mask : ``torch.Tensor``, (batch_size, trimmed_source_length)

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor], (batch_size,), (batch_size, max_input_sequence_length)
        """
        batch_size, trimmed_source_length = copy_mask.size()

        # We use the same trick of subtracting the max before taking the exp
        # that is used in the logsumexp function.
        max_score, _ = torch.cat((generation_scores, copy_scores), dim=-1).max(-1)

        stable_gen_scores = generation_scores - max_score.unsqueeze(-1)
        stable_gen_exp_scores = stable_gen_scores.exp()
        # shape: (batch_size, target_vocab_size)

        stable_copy_scores = copy_scores - max_score.unsqueeze(-1)
        stable_copy_exp_scores = stable_copy_scores.exp() * copy_mask
        # shape: (batch_size, trimmed_source_length)

        # This mask ensures that we only add generation scores for targets that are
        # in the target vocabulary or not in the source sentence
        # (in which case it would be the OOV token score).
        gen_mask = ((target_tokens != self._oov_index) | (source_indices.sum(-1) == 0)).float()
        # shape: (batch_size,)

        stable_gen_exp = stable_gen_exp_scores.gather(1, target_tokens.unsqueeze(1)).squeeze(-1) * gen_mask
        # shape: (batch_size,)

        stable_copy_exp_scores_filtered = stable_copy_exp_scores * source_indices.float()
        # shape: (batch_size, trimmed_source_length)

        stable_copy_exp = stable_copy_exp_scores_filtered.sum(dim=-1)
        # shape: (batch_size,)

        # NOTE: we omit adding `max_score` back on since it would be cancelled
        # out below anyway.
        normalization = (stable_gen_exp_scores.sum(-1) + stable_copy_exp_scores.sum(-1)).log()
        # shape: (batch_size,)

        selective_weights = (stable_copy_exp_scores_filtered.log() -
                             normalization.unsqueeze(-1).expand(batch_size, trimmed_source_length)).exp()
        # shape: (batch_size, trimmed_source_length)

        step_log_likelihood = (stable_gen_exp + stable_copy_exp).log() - normalization
        # shape: (batch_size,)

        return step_log_likelihood, selective_weights

    def _forward_train(self,
                       target_tokens: Dict[str, torch.LongTensor],
                       source_indices: torch.Tensor,
                       state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        source_mask = state["source_mask"]
        # shape: (batch_size, max_input_sequence_length)

        batch_size, target_sequence_length = target_tokens["tokens"].size()

        # The last input from the target is either padding or the end symbol.
        # Either way, we don't have to process it.
        num_decoding_steps = target_sequence_length - 1

        # We initialize the target predictions with the start token.
        input_choices = source_mask.new_full((batch_size,), fill_value=self._start_index)
        # shape: (batch_size,)

        # We use this to fill in the copy index when the previous input was copied.
        copy_input_choices = source_mask.new_full((batch_size,), fill_value=self._copy_index)
        # shape: (batch_size,)

        copy_mask = source_mask[:, 1:-1].float()
        # shape: (batch_size, trimmed_source_length)

        # We need to keep track of the probabilities assigned to tokens in the source
        # sentence that were copied during the previous timestep, since we use
        # those probabilities as weights when calculating the "selective read".
        selective_weights = state["decoder_hidden"].new_zeros(copy_mask.size())
        # shape: (batch_size, trimmed_source_length)

        log_likelihood = state["decoder_hidden"].new_zeros((batch_size,))
        # shape: batch_size,)

        for timestep in range(num_decoding_steps):
            input_choices = target_tokens["tokens"][:, timestep]
            # shape: (batch_size,)

            # If the previous target token was copied, we use the special copy token.
            # But the end target token will always be THE end token, so we know
            # it was not copied.
            if timestep < num_decoding_steps - 1:
                # Get mask tensor indicating which instances were copied.
                copied = (source_indices[:, timestep, :].sum(-1) > 0).long()
                # shape: (batch_size,)

                input_choices = input_choices * (1 - copied) + copy_input_choices * copied
                # shape: (batch_size,)

            # Update the decoder state by taking a step through the RNN.
            state = self._decoder_step(input_choices, selective_weights, state)

            # Get generation scores for each token in the target vocab.
            generation_scores = self._get_generation_scores(state)
            # (batch_size, target_vocab_size)

            # Get copy scores for each token in the source sentence, excluding the start
            # and end tokens.
            copy_scores = self._get_copy_scores(state)
            # (batch_size, max_input_sequence_length - 2)

            step_target_tokens = target_tokens["tokens"][:, timestep + 1]
            # shape: (batch_size,)

            step_source_indices = source_indices[:, timestep + 1]
            # shape: (batch_size, max_input_sequence_length - 2)

            step_log_likelihood, selective_weights = self._get_ll_contrib(generation_scores,
                                                                          copy_scores,
                                                                          step_target_tokens,
                                                                          step_source_indices,
                                                                          copy_mask)
            log_likelihood = log_likelihood + step_log_likelihood

        loss = - log_likelihood.sum() / batch_size

        return {"loss": loss}

    def _forward_predict(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError
