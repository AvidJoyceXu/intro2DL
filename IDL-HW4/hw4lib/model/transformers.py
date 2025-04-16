import torch.nn as nn
import torch
import random
from typing import Tuple, Optional, Literal
from .masks import PadMask, CausalMask
from .positional_encoding import PositionalEncoding
from .decoder_layers import SelfAttentionDecoderLayer, CrossAttentionDecoderLayer
from .encoder_layers import SelfAttentionEncoderLayer
from .speech_embedding import SpeechEmbedding
import warnings
from torchinfo import summary
'''
TODO: Implement these Modules.

This file contains two key transformer architectures:

1. DecoderOnlyTransformer: Used for language modeling tasks (like GPT)
   - Contains a stack of SelfAttentionDecoderLayers
   - Uses causal masking to prevent attending to future tokens
   - Includes optional weight tying and layer dropout features

    Key components to implement:
    1. Token Embedding Layer: Convert token IDs to vectors
    2. Positional Encoding: Add position information
    3. Decoder Stack: Process tokens sequentially
    4. Output Projection: Convert final representations to logits

    Architecture follows Pre-LN (Layer Normalization) design where:
    - Layer normalization is applied at the start of each sublayer
    - Residual connections wrap around each sublayer
    - Final layer norm is applied before output projection

    Implementation Notes:
    1. The forward pass should handle:
    - Proper masking (both padding and causal)
    - Collecting attention weights from all layers
    - Optional layer dropout during training
    
    2. The score method should:
    - Handle single token prediction
    - Not apply padding masks
    - Return only the final token's logits

2. EncoderDecoderTransformer: Used for ASR (Automatic Speech Recognition) tasks
   - Contains an encoder stack for processing speech features
   - Contains a decoder stack for generating text tokens
   - Uses both self-attention and cross-attention mechanisms
   - Includes CTC auxiliary loss support and optional weight tying

   Key components to implement:
   1. Speech Embedding: Convert speech features to vectors with time reduction
   2. Positional Encoding: Add position information (optional for both encoder/decoder)
   3. Encoder Stack: Process speech features
   4. Decoder Stack: Generate text tokens
   5. CTC Head: For auxiliary CTC loss computation
   6. Output Projection: Convert final representations to logits

   Architecture follows Pre-LN (Layer Normalization) design where:
   - Layer normalization is applied at the start of each sublayer
   - Residual connections wrap around each sublayer
   - Final layer norm is applied before output projection

   Implementation Notes:
   1. The forward pass should handle:
   - Proper masking (padding for encoder, both padding and causal for decoder)
   - Collecting attention weights from all layers
   - Optional layer dropout during training
   - CTC logits computation

   2. The score method should:
   - Handle single token prediction given encoder output
   - Not apply padding masks to decoder inputs
   - Return only the final token's logits
'''

## -------------------------------------------------------------------------------------------------
## Decoder-Only Transformer
## -------------------------------------------------------------------------------------------------
class DecoderOnlyTransformer(nn.Module):
    '''
    A Pre-LN Decoder-Only Transformer model.
    '''
    def __init__(
            self, 
            num_layers: int, 
            d_model: int, 
            num_heads: int, 
            d_ff: int, 
            dropout: float, 
            max_len: int, 
            num_classes: int,
            weight_tying: bool = False,
            layer_drop_rate: float = 0.0,
    ):
        '''
        Initialize the Decoder-Only Transformer model.

        Args:
            num_layers: int, number of decoder layers
            d_model: int, model dimension
            num_heads: int, number of attention heads
            d_ff: int, feed-forward dimension
            dropout: float, dropout rate
            max_len: int, maximum sequence length this model can handle
            num_classes: int, number of classes
            weight_tying: bool, whether to use weight tying (default: False)
            layer_drop_rate: float, layer drop rate (default: 0.0)
        '''
        super().__init__()
        
        # TODO: Implement __init__

        # Initialize the decoder
        # DO NOT MODIFY THESE ATTRIBUTES
        self.max_len         = max_len
        self.layer_drop_rate = layer_drop_rate
        self.num_classes     = num_classes
        self.num_layers      = num_layers
        
        # TODO: Create a ModuleList of decoder layers based on the number of layers
        self.dec_layers     = NotImplementedError # ModuleList of decoder layers

        # TODO: Create target embedding and other layers
        self.target_embedding       = NotImplementedError # Target embedding
        self.positional_encoding    = NotImplementedError # Positional encoding
        self.final_linear           = NotImplementedError # Final linear layer
        self.dropout                = NotImplementedError # Dropout
        self.norm                   = NotImplementedError # Layer norm

        # Weight tying (extra form of regularization, read more about it)
        if weight_tying:
            self.target_embedding.weight = self.final_linear.weight

        raise NotImplementedError # Remove once implemented

    def forward(self, padded_targets: torch.Tensor, target_lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, dict]:
        '''
        Forward pass for the decoder. Used for Training only. Tokens are assumed to be right-padded.
        Args:
            padded_targets (torch.Tensor): The padded target sequence. shape: (batch_size, seq_len)
            target_lengths (Optional[torch.Tensor]): The lengths of the target sequences. shape: (batch_size,)
        Returns:
            seq_out (torch.Tensor): The output sequence. shape: (batch_size, seq_len, d_model)
            runnint_att (dict): The attention weights. shape: (batch_size, seq_len, seq_len)
        '''
        # DO NOT MODIFY 
        if self.training and target_lengths is None:
            raise ValueError("target_lengths must be provided during training")
        
        # TODO: Implement forward

        # TODO: Create padding mask for padded_targets on the same device as the input (use PadMask)
        pad_mask_dec = None
        if target_lengths is not None:
            pad_mask_dec = NotImplementedError
        
        # TODO: Create causal mask to prevent attending to future tokens on the same device as the input (use CausalMask)
        causal_mask = NotImplementedError

        # TODO: Apply the embedding
        x = NotImplementedError
        # TODO: Apply positional encoding
        x = NotImplementedError
        # TODO: Apply dropout 
        x = NotImplementedError

        # TODO: Pass through all decoder layers, save attention masks
        runnint_att = {}
        for i in range(self.num_layers):
            # Optionally apply LayerDrop during training (More regularization!)
            if self.training and self.layer_drop_rate > 0 and random.random() < self.layer_drop_rate:
                continue
            
            # TODO: Pass through decoder layer
            x, attention = NotImplementedError, NotImplementedError
            
            # TODO: Save attention weights  
            runnint_att['layer{}_dec_self'.format(i + 1)] = attention

        # TODO: Apply normalization
        x = NotImplementedError
        # TODO: Linear layer (Final Projection) for next character prediction
        seq_out = NotImplementedError
        
        # TODO: Return the output sequence and running attention weights
        raise NotImplementedError
    
    def score(self, batch_prompts: torch.Tensor) -> torch.Tensor:
        '''
        Score the tokens for the decoder. 
        This is used for scoring the next token for a given prompt.
        Padding mask is not applied so ensure that the prompts are not padded. 
        Can only handle batch_size = 1 or batch with same lengths and no padding. 
        Args:
            prompts (torch.Tensor) : tensor of fixed length token sequences. shape: (batch_size, seq_len)
        Returns:
            logits (torch.Tensor): Batch of next token logits. shape: (batch_size, num_classes)
        '''
        if self.training:
            raise ValueError("score method is not supported during training, use forward method instead")
        # Forward pass with no target lengths
        seq_out, _ = self.forward(batch_prompts, target_lengths=None)
        # Return the last token's logits for next token prediction    
        logits     = seq_out[:, -1, :]
        return logits
    

## -------------------------------------------------------------------------------------------------
## Encoder-Decoder Transformer
## -------------------------------------------------------------------------------------------------
class EncoderDecoderTransformer(nn.Module):
    '''
    A Pre-LN Encoder-Decoder Transformer model for ASR tasks.
    '''
    def __init__(
            self,
            input_dim: int,  
            time_reduction: int, 
            reduction_method: Literal['lstm', 'conv', 'both'], 
            num_encoder_layers: int,
            num_encoder_heads: int,
            d_ff_encoder: int, 
            num_decoder_layers: int,
            num_decoder_heads: int,
            d_ff_decoder: int,
            d_model: int,
            dropout: float, 
            max_len: int, 
            num_classes: int,
            weight_tying: bool = False,
            layer_drop_rate: float = 0.0,
            skip_encoder_pe: bool = False,
            skip_decoder_pe: bool = False,
    ):
        '''
        Initialize the Encoder-Decoder Transformer model.

        Args:
            input_dim: int, dimension of input speech features
            time_reduction: int, stride along time dimension, the amount of reduction to apply to the time dimension
            reduction_method: Literal['lstm', 'conv', 'both'], the source_embedding reduction method
            num_encoder_layers: int, number of encoder layers
            num_encoder_heads: int, number of encoder attention heads
            d_ff_encoder: int, feed-forward dimension for encoder
            num_decoder_layers: int, number of decoder layers
            num_decoder_heads: int, number of decoder attention heads
            d_ff_decoder: int, feed-forward dimension for decoder
            d_model: int, model dimension
            dropout: float, dropout rate
            max_len: int, maximum sequence length this model can handle
            num_classes: int, number of classes
            weight_tying: bool, whether to use weight tying (default: False)
            layer_drop_rate: float, layer drop rate (default: 0.0)
            skip_encoder_pe: bool, whether to skip positional encoding for encoder (default: False)
            skip_decoder_pe: bool, whether to skip positional encoding for decoder (default: False)
        '''
        super().__init__()

        # TODO: Implement __init__

        # Initialize model attributes
        # DO NOT MODIFY THESE ATTRIBUTES
        self.max_len = max_len
        self.layer_drop_rate = layer_drop_rate
        self.num_classes = num_classes
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.skip_encoder_pe = skip_encoder_pe
        self.skip_decoder_pe = skip_decoder_pe

        # TODO: Create encoder layers
        # Use ModuleList to create a list of encoder layers
        self.enc_layers = NotImplementedError # ModuleList of encoder layers

        # TODO: Create decoder layers
        # Use ModuleList to create a list of decoder layers
        self.dec_layers = NotImplementedError # ModuleList of decoder layers

        # TODO: Create source and target embeddings and other layers
        # Use SpeechEmbedding class to create the source embedding
        self.source_embedding = NotImplementedError # Speech embedding


        # TODO: Create the target embedding
        # Use nn.Embedding class to create the target embedding
        self.target_embedding    = NotImplementedError # Target embedding

        # TODO: Create the positional encoding layer
        self.positional_encoding = NotImplementedError # Positional encoding

        # TODO: Create the final linear layer
        self.final_linear        = NotImplementedError # Final linear layer

        # TODO: Create the dropout layer
        self.dropout             = NotImplementedError # Dropout

        # TODO: Create the encoder normalization layer
        self.encoder_norm        = NotImplementedError # Encoder normalization

        # TODO: Create the decoder normalization layer
        self.decoder_norm        = NotImplementedError # Decoder normalization

        # TODO: Create the CTC head
        # Use nn.Sequential to create the CTC head
        # CTC head should project the final encoder output from the d_model space to the num_classes space
        # To be compatible with CTCLoss, a log_softmax to the output (See. nn.LogSoftmax)
        self.ctc_head            = NotImplementedError # CTC head


        # Weight tying if enabled (extra form of regularization, read more about it)
        if weight_tying:
            self.target_embedding.weight = self.final_linear.weight

        raise NotImplementedError # Remove once implemented

    def encode(self, padded_sources: torch.Tensor, source_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        '''
        Encodes the source features into a sequence of hidden states.
        Args:
            padded_sources: The padded source sequences. shape: (batch_size, src_len, input_dim)
            source_lengths: The lengths of source sequences. shape: (batch_size,)
        Returns:
            x_enc: Encoded representation. shape: (batch_size, src_len, d_model)
            pad_mask_src: Source padding mask. shape: (batch_size, src_len)
            running_att: Dictionary containing encoder self-attention weights
            ctc_inputs: Dictionary of CTC input and source lengths. shape: (src_len, batch_size, d_model), (batch_size,) 
                        Keys: 'log_probs' and 'lengths'
                        Required for CTC loss computation
        '''

        # TODO: Implement encode

        # TODO: Apply speech embedding
        x_enc, x_enc_lengths = NotImplementedError, NotImplementedError
        
        # TODO: Apply positional encoding if not skipped
        # You can try to optionally skipping positional encoding if using an LSTM based speech embedding
        # LSTM embeddings on their own can be sufficient to capture the positional information    
        if not self.skip_encoder_pe:
            x_enc = NotImplementedError
        
        # TODO: Apply dropout
        x_enc = NotImplementedError

        # TODO: Create source padding mask on the same device as the input
        pad_mask_src = NotImplementedError

        # TODO: Pass through encoder layers and save attention weights
        running_att = {}
        for i in range(self.num_encoder_layers):
            # Optionally apply LayerDrop during training (More regularization!)
            if self.training and self.layer_drop_rate > 0 and random.random() < self.layer_drop_rate:
                continue
            # TODO: Pass through encoder layer
            x_enc, attention = NotImplementedError, NotImplementedError
            
            # Save attention weights
            running_att[f'layer{i+1}_enc_self'] = attention

        # TODO: Apply normalization
        x_enc = NotImplementedError
        # TODO: Project to CTC logits
        ctc_logits = NotImplementedError

        # TODO: Return the encoded representation, padding mask, running attention weights, and CTC inputs (see docstring)
        raise NotImplementedError

    def decode(
        self, 
        padded_targets: torch.Tensor, 
        encoder_output: torch.Tensor,
        target_lengths: Optional[torch.Tensor] = None,
        pad_mask_src: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        '''
        Decode the target sequence conditioned on the encoder output.
        Args:
            padded_targets: The padded target sequence. shape: (batch_size, tgt_len)
            encoder_output: Output from encoder. shape: (batch_size, src_len, d_model)
            target_lengths: The lengths of target sequences. shape: (batch_size,)
            pad_mask_src: Source padding mask from encoder. shape: (batch_size, src_len)
        Returns:
            seq_out: The output sequence. shape: (batch_size, tgt_len, num_classes)
            running_att: Dictionary containing decoder attention weights
        '''
        # TODO: Implement decode

        # TODO: Create target padding mask on the same device as the input
        pad_mask_tgt = None
        if target_lengths is not None:
            pad_mask_tgt = NotImplementedError

        if pad_mask_tgt is None and self.training:
            warnings.warn("pad_mask_tgt is None, unless you are using the decoder as a standalone model or doing inference, you should provide target_lengths")

        # TODO: Create causal mask on the same device as the input
        causal_mask = NotImplementedError

        # TODO: Apply the embedding, positional encoding, and dropout
        x_dec = NotImplementedError

        # TODO: Apply positional encoding if not skipped
        # Shouldn't really be doing this. Included for completeness.  
        if not self.skip_decoder_pe:
            x_dec = NotImplementedError

        # TODO: Apply dropout
        x_dec = NotImplementedError

        # TODO: Pass through decoder layers and save attention weights
        running_att = {}
        for i in range(self.num_decoder_layers):
            if self.training and self.layer_drop_rate > 0 and random.random() < self.layer_drop_rate:
                continue
            # TODO: Pass through decoder layer
            x_dec, self_attn, cross_attn = NotImplementedError, NotImplementedError, NotImplementedError
            
            # TODO: Save attention weights
            running_att[f'layer{i+1}_dec_self'] = self_attn
            running_att[f'layer{i+1}_dec_cross'] = cross_attn

        # TODO: Final normalization
        x_dec = NotImplementedError

        # TODO: Final projection
        seq_out = NotImplementedError

        # TODO: Return the output sequence and running attention weights
        raise NotImplementedError

    def forward(
        self,
        padded_sources: torch.Tensor,
        padded_targets: torch.Tensor,
        source_lengths: Optional[torch.Tensor] = None,
        target_lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        '''
        Forward pass for the encoder-decoder transformer.
        
        Args:
            padded_sources: The padded source sequences. shape: (batch_size, src_len, input_dim)
            padded_targets: The padded target sequences. shape: (batch_size, tgt_len)
            source_lengths: The lengths of source sequences. shape: (batch_size,)
            target_lengths: The lengths of target sequences. shape: (batch_size,)
            
        Returns:
            seq_out: The output sequence logits. shape: (batch_size, tgt_len, num_classes)
            running_att: Dictionary containing all attention weights from both encoder and decoder
            ctc_inputs: Dictionary of CTC input and source lengths. shape: (src_len, batch_size, d_model), (batch_size,) 
                        Keys: 'log_probs' and 'lengths'
                        Required for CTC loss computation
        '''
        # During training, we need target lengths
        if self.training and target_lengths is None:
            raise ValueError("target_lengths must be provided during training")

        if self.training and source_lengths is None:
            raise ValueError("source_lengths must be provided during training")
        
        # TODO: Implement forward

        # TODO: Encode the source sequence
        encoder_output, pad_mask_src, enc_running_att, ctc_inputs = NotImplementedError, NotImplementedError, NotImplementedError, NotImplementedError
        
        # TODO: Decode using encoder output
        seq_out, dec_running_att = NotImplementedError, NotImplementedError
        
        # Combine attention dictionaries
        running_att = {**enc_running_att, **dec_running_att}
        
        # TODO: Return the output sequence, running attention weights, and CTC inputs (see docstring)
        raise NotImplementedError

    def score(self, batch_prompts: torch.Tensor, encoder_output: torch.Tensor, pad_mask_src: torch.Tensor) -> torch.Tensor:
        '''
        Score the next token for given encoder output and prompt.
        Args:
            batch_prompts: tensor of token sequences to score for next token. shape: (batch_size, seq_len)
            encoder_output: encoder output/hidden states. shape: (batch_size, src_len, d_model)
            pad_mask_src: source padding mask. shape: (batch_size, src_len)
        Returns:
            logits: Batch of next token logits. shape: (batch_size, num_classes)
        '''
        if self.training:
            raise ValueError("score method is not supported during training")

        # TODO: Use decode function with no target lengths (no padding mask for targets)
        seq_out, _ = self.decode(batch_prompts, encoder_output, None, pad_mask_src)
        
        # Return only the last token's logits
        return seq_out[:, -1, :]


    @classmethod
    def from_pretrained_decoder(
        cls,
        decoder_checkpoint_path: str,
        config: dict,
    ) -> Tuple['EncoderDecoderTransformer', dict]:
        """
        Helper function to initialize an encoder-decoder transformer with decoder weights initialized from a pretrained decoder-only model.
        
        Args:
            decoder_checkpoint_path: Path to decoder-only transformer checkpoint
            config: Configuration dictionary for the encoder-decoder model
            
        Returns:
            model: Initialized encoder-decoder transformer
            param_info: Dictionary containing lists of named parameters {'transferred': [(name, param)], 'new': [(name, param)]}
        """
        print("\n=== Initializing Encoder-Decoder from Pretrained Decoder ===")
        print(f"Loading checkpoint from: {decoder_checkpoint_path}")
        
        # Create new encoder-decoder model
        print("\nCreating new encoder-decoder model...")
        model = cls(**config)

        # Load decoder checkpoint
        print("Loading pretrained decoder weights...")
        checkpoint = torch.load(decoder_checkpoint_path, map_location='cpu', weights_only=True)
        decoder_state_dict = checkpoint['model_state_dict']
        
        # Track named parameters
        transferred_params = []
        new_params = []
        
        def transfer_module_weights(target_module, prefix):
            module_state_dict = {
                k.replace(prefix, ''): v 
                for k, v in decoder_state_dict.items()
                if k.startswith(prefix)
            }
            param_count = sum(p.numel() for p in target_module.parameters())
            print(f"  - Transferring {prefix} ({param_count:,} parameters)")
            target_module.load_state_dict(module_state_dict)
            # Store the full parameter names with their prefix
            for name, param in target_module.named_parameters():
                transferred_params.append((f"{prefix}{name}", param))

        # Transfer shared components
        print("\nTransferring shared components:")
        transfer_module_weights(model.target_embedding, 'target_embedding.')
        transfer_module_weights(model.final_linear, 'final_linear.')
        transfer_module_weights(model.decoder_norm, 'norm.')
        
        # Transfer decoder layers
        num_layers = min(
            len([k for k in decoder_state_dict.keys() if k.startswith('dec_layers.')]) // 2,
            model.num_decoder_layers
        )
        print(f"\nTransferring decoder layers (found {num_layers} layers):")
        
        for i in range(num_layers):
            print(f"\nLayer {i + 1}/{num_layers}:")
            transfer_module_weights(
                model.dec_layers[i].self_attn,
                f'dec_layers.{i}.self_attn.'
            )
            transfer_module_weights(
                model.dec_layers[i].ffn,
                f'dec_layers.{i}.ffn.'
            )
        
        # Collect new parameters with their names
        print("\nCollecting new parameters...")
        for name, param in model.named_parameters():
            is_new = True
            for transferred_name, transferred_param in transferred_params:
                if param is transferred_param:
                    is_new = False
                    break
            if is_new:
                new_params.append((name, param))
        
        print("\n=== Initialization Complete ===")
        return model, {'transferred': transferred_params, 'new': new_params}

    def log_param_groups(self, param_groups: list) -> None:
        """Log information about parameter groups."""
        print("\nParameter groups:")
        total_params = 0
        total_trainable = 0
        
        for group in param_groups:
            num_params = sum(p.numel() for p in group['params'])
            trainable = sum(p.numel() for p in group['params'] if p.requires_grad)
            total_params += num_params
            total_trainable += trainable
            
            print(f"\n{group['name']}:")
            print(f"  Parameters: {num_params:,}")
            print(f"  Trainable: {trainable:,}")
            print(f"  LR factor: {group['lr_factor']}")
        
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Total trainable: {total_trainable:,}")


## -------------------------------------------------------------------------------------------------
## Test Cases
## -------------------------------------------------------------------------------------------------

def get_decoder_only_inputs(max_len: int = 300, num_classes: int = 10000):
    batch_size = 8
    padded_targets = torch.randint(0, num_classes, (batch_size, max_len))
    source_lengths = torch.ones(batch_size) * max_len
    return padded_targets, source_lengths


def get_encoder_decoder_inputs(max_len: int = 300, num_classes: int = 10000):
    batch_size = 8
    padded_targets = torch.randint(0, num_classes, (batch_size, max_len))
    source_lengths = torch.ones(batch_size) * max_len
    return padded_targets, source_lengths


def test_decoder_only(num_layers: int = 12, num_heads: int = 8, d_model: int = 512, d_ff: int = 2048, dropout: float = 0.1, max_len: int = 300, num_classes: int = 1000):
    padded_targets, target_lengths = get_decoder_only_inputs(max_len, num_classes)
    model = DecoderOnlyTransformer(num_layers, d_model, num_heads, d_ff, dropout, max_len, num_classes)
    summary(model, input_data=[padded_targets, target_lengths])

if __name__ == "__main__":
    test_decoder_only()