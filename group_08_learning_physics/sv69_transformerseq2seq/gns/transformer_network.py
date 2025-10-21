import torch
import torch.nn as nn
import math
from typing import List
import copy
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from torch.nn import functional as F


class TransformerProcessor(nn.Module):
    """Transformer-based processor replacing the message-passing GNN."""

    def __init__(self, nnode_in: int,
                    nnode_out: int,
                    nedge_in: int,
                    nedge_out: int,
                    nmessage_passing_steps: int,
                    nmlp_layers: int,
                    mlp_hidden_dim: int,
                    n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=nnode_in,
            nhead=n_heads,
            dim_feedforward=nnode_in * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=nmessage_passing_steps)
        self.layernorm = nn.LayerNorm(nnode_in)


    def forward(self, x: torch.Tensor, edge_index=None, edge_features=None, positions: torch.Tensor = None) -> torch.Tensor:
        x = x.unsqueeze(0)  # (1, nparticles, latent_dim)
        x = self.transformer(x)
        x = self.layernorm(x)
        return x.squeeze(0), edge_features
    
class PhysicsPositionalEncoding(nn.Module):
    def __init__(self, num_dimensions: int, num_frequencies: int = 16, embed_dim: int = 128):
        super().__init__()
        self.num_dimensions = num_dimensions
        self.num_frequencies = num_frequencies
        self.embed_dim = embed_dim

        self.freq_bands = 2.0 ** torch.linspace(0, num_frequencies - 1, num_frequencies)

        # Final linear projection to match model dimension
        self.linear = nn.Linear(num_dimensions * num_frequencies * 2, embed_dim)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            positions: tensor of shape (N, num_dimensions) with absolute coordinates
        Returns:
            pos_embed: tensor of shape (N, embed_dim)
        """
        # Expand with frequencies: (N, num_dimensions, num_frequencies)
        angles = positions.unsqueeze(-1) * self.freq_bands.to(positions.device)

        sin_feats = torch.sin(angles)
        cos_feats = torch.cos(angles)
        fourier_features = torch.cat([sin_feats, cos_feats], dim=-1).view(positions.size(0), -1)

        pos_embed = self.linear(fourier_features)
        return pos_embed

class PhysicsTransformerProcessorRadiusMasked(nn.Module):
    """
    Transformer processor for particle systems with local (radius-based) attention masking.

    Each particle can only attend to other particles within a cutoff distance,
    similar to the neighbor graph in GNN-based simulators.
    """

    def __init__(
        self,
        nnode_in: int,
        nnode_out: int,
        nedge_in: int,
        nedge_out: int,
        nmessage_passing_steps: int,
        nmlp_layers: int,
        mlp_hidden_dim: int,
        num_dimensions: int = 2,
        n_heads: int = 8,
        dropout: float = 0.1,
        num_frequencies: int = 16,
        radius: float = 0.02, 
    ):
        super().__init__()
        self.radius = radius
        self.n_heads = n_heads

        # Positional encoder
        self.pos_encoder = PhysicsPositionalEncoding(
            num_dimensions=num_dimensions,
            num_frequencies=num_frequencies,
            embed_dim=nnode_in,
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=nnode_in,
            nhead=n_heads,
            dim_feedforward=nnode_in * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=copy.deepcopy(encoder_layer),
            num_layers=nmessage_passing_steps,
            norm=nn.LayerNorm(nnode_in),
        )

        self.dropout = nn.Dropout(dropout)

    def build_radius_mask(self, positions: torch.Tensor, radius: float) -> torch.Tensor:
        """
        Build attention mask using Euclidean distance (radius cutoff).

        Returns:
            mask: (N, N) BoolTensor where True = blocked attention
        """
        dist = torch.cdist(positions, positions)  # (N, N)
        mask = dist > radius
        mask.fill_diagonal_(False)
        return mask

    def forward(self, node_features, edge_index=None, edge_features=None, positions=None):
        """
        Args:
            node_features: (N, latent_dim)
            positions: (N, num_dimensions)
        Returns:
            updated_features: (N, latent_dim)
        """
        num_nodes = node_features.size(0)
        pos_embed = self.pos_encoder(positions)
        x = node_features + self.dropout(pos_embed)
        x = x.unsqueeze(0) 

        # Compute radius-based attention mask
        attn_mask = self.build_radius_mask(positions, self.radius)

        # Forward transformer with locality mask
        x = self.transformer(x, mask=attn_mask)
        x = x.squeeze(0)

        return x, edge_features


class PhysicsTransformerProcessor(nn.Module):
    """
    Transformer model that updates particle embeddings
    using attention, conditioned on their absolute positions.
    """

    def __init__(
        self,
        nnode_in: int,
        nnode_out: int,
        nedge_in: int,
        nedge_out: int,
        nmessage_passing_steps: int,
        nmlp_layers: int,
        mlp_hidden_dim: int,

        num_dimensions: int = 2,
        n_heads: int = 8,
        dropout: float = 0.1,
        num_frequencies: int = 16,
    ):
        super().__init__()

        # Positional encoder that maps absolute 3D coords → embedding
        self.pos_encoder = PhysicsPositionalEncoding(
            num_dimensions=num_dimensions,
            num_frequencies=num_frequencies,
            embed_dim=nnode_in,
        )
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=nnode_in,
            nhead=n_heads,
            dim_feedforward=nnode_in * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=copy.deepcopy(encoder_layer),
            num_layers=nmessage_passing_steps,
            norm=nn.LayerNorm(nnode_in),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, node_features: torch.Tensor, edge_index=None, edge_features=None, positions: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            node_features: (N, latent_dim) per-particle latent features
            positions: (N, num_dimensions) absolute particle coordinates
        Returns:
            updated_features: (N, latent_dim) updated node embeddings
        """
        # Add continuous positional encoding based on absolute spatial location
        pos_embed = self.pos_encoder(positions)
        x = node_features + self.dropout(pos_embed)

        # Add batch dimension for transformer (batch_first=True)
        x = x.unsqueeze(0)  # (1, N, latent_dim)
        x = self.transformer(x)
        x = x.squeeze(0)  # (N, latent_dim)
        return x, edge_features  # passthrough edge features


def build_mlp(
        input_size: int,
        hidden_layer_sizes: List[int],
        output_size: int = None,
        output_activation: nn.Module = nn.Identity,
        activation: nn.Module = nn.ReLU) -> nn.Module:
  """Build a MultiLayer Perceptron.

  Args:
    input_size: Size of input layer.
    layer_sizes: An array of input size for each hidden layer.
    output_size: Size of the output layer.
    output_activation: Activation function for the output layer.
    activation: Activation function for the hidden layers.

  Returns:
    mlp: An MLP sequential container.
  """
  # Size of each layer
  layer_sizes = [input_size] + hidden_layer_sizes
  if output_size:
    layer_sizes.append(output_size)

  # Number of layers
  nlayers = len(layer_sizes) - 1

  # Create a list of activation functions and
  # set the last element to output activation function
  act = [activation for i in range(nlayers)]
  act[-1] = output_activation

  # Create a torch sequential container
  mlp = nn.Sequential()
  for i in range(nlayers):
    mlp.add_module("NN-" + str(i), nn.Linear(layer_sizes[i],
                                             layer_sizes[i + 1]))
    mlp.add_module("Act-" + str(i), act[i]())

  return mlp


class Encoder(nn.Module):
  """Graph network encoder. Encode nodes and edges states to an MLP. The Encode:
  :math: `\mathcal{X} \rightarrow \mathcal{G}` embeds the particle-based state
  representation, :math: `\mathcal{X}`, as a latent graph, :math:
  `G^0 = encoder(\mathcal{X})`, where :math: `G = (V, E, u), v_i \in V`, and
  :math: `e_{i,j} in E`
  """

  def __init__(
          self,
          nnode_in_features: int,
          nnode_out_features: int,
          nedge_in_features: int,
          nedge_out_features: int,
          nmlp_layers: int,
          mlp_hidden_dim: int):
    """The Encoder implements nodes features :math: `\varepsilon_v` and edge
    features :math: `\varepsilon_e` as multilayer perceptrons (MLP) into the
    latent vectors, :math: `v_i` and :math: `e_{i,j}`, of size 128.

    Args:
      nnode_in_features: Number of node input features (for 2D = 30, calculated
        as [10 = 5 times steps * 2 positions (x, y) +
        4 distances to boundaries (top/bottom/left/right) +
        16 particle type embeddings]).
      nnode_out_features: Number of node output features (latent dimension of
        size 128).
      nedge_in_features: Number of edge input features (for 2D = 3, calculated
        as [2 (x, y) relative displacements between 2 particles + distance
        between 2 particles]).
      nedge_out_features: Number of edge output features (latent dimension of
        size 128).
      nmlp_layer: Number of hidden layers in the MLP (typically of size 2).
      mlp_hidden_dim: Size of the hidden layer (latent dimension of size 128).

    """
    super(Encoder, self).__init__()
    # Encode node features as an MLP
    self.node_fn = nn.Sequential(*[build_mlp(nnode_in_features,
                                             [mlp_hidden_dim
                                              for _ in range(nmlp_layers)],
                                             nnode_out_features),
                                   nn.LayerNorm(nnode_out_features)])
    # # Encode edge features as an MLP
    # self.edge_fn = nn.Sequential(*[build_mlp(nedge_in_features,
    #                                          [mlp_hidden_dim
    #                                           for _ in range(nmlp_layers)],
    #                                          nedge_out_features),
    #                                nn.LayerNorm(nedge_out_features)])

  def forward(
          self,
          x: torch.tensor,
          edge_features: torch.tensor):
    """The forward hook runs when the Encoder class is instantiated

    Args:
      x: Particle state representation as a torch tensor with shape
        (nparticles, nnode_input_features)
      edge_features: Edge features as a torch tensor with shape
        (nparticles, nedge_input_features)

    """
    # return self.node_fn(x), self.edge_fn(edge_features)
    return self.node_fn(x), edge_features  # Passthrough edge features


class Processor(nn.Module):
  """The Processor: :math: `\mathcal{G} \rightarrow \mathcal{G}` computes 
  interactions among nodes via :math: `M` steps of learned message-passing, to 
  generate a sequence of updated latent graphs, :math: `G = (G_1 , ..., G_M )`, 
  where :math: `G^{m+1| = GN^{m+1} (G^m )`. It returns the final graph, 
  :math: `G^M = PROCESSOR(G^0)`. Message-passing allows information to 
  propagate and constraints to be respected: the number of message-passing 
  steps required will likely scale with the complexity of the interactions.

  """

  def __init__(
      self,
      nnode_in: int,
      nnode_out: int,
      nedge_in: int,
      nedge_out: int,
      nmessage_passing_steps: int,
      nmlp_layers: int,
      mlp_hidden_dim: int,
  ):
    """Processor derived from torch_geometric MessagePassing class. The 
    processor uses a stack of :math: `M GNs` (where :math: `M` is a 
    hyperparameter) with identical structure, MLPs as internal edge and node 
    update functions, and either shared or unshared parameters. We use GNs 
    without global features or global updates (i.e., an interaction network), 
    and with a residual connections between the input and output latent node 
    and edge attributes.

    Args:
      nnode_in: Number of node inputs (latent dimension of size 128).
      nnode_out: Number of node outputs (latent dimension of size 128).
      nedge_in: Number of edge inputs (latent dimension of size 128).
      nedge_out: Number of edge output features (latent dimension of size 128).
      nmessage_passing_steps: Number of message passing steps.
      nmlp_layer: Number of hidden layers in the MLP (typically of size 2).
      mlp_hidden_dim: Size of the hidden layer (latent dimension of size 128).

    """
    super(Processor, self).__init__(aggr='max')
    # Create a stack of M Graph Networks GNs.
    self.gnn_stacks = nn.ModuleList([
        TransformerProcessor(
            latent_dim=nnode_in,
            n_heads=8,
            n_layers=nmessage_passing_steps,
            dropout=0.1
        ) for _ in range(1)])

  def forward(self,
              x: torch.tensor,
              edge_index: torch.tensor,
              edge_features: torch.tensor,positions: torch.tensor):
    """The forward hook runs through GNN stacks when class is instantiated. 

    Args:
      x: Particle state representation as a torch tensor with shape 
        (nparticles, latent_dim)
      edge_index: A torch tensor list of source and target nodes with shape 
        (2, nedges)
      edge_features: Edge features as a torch tensor with shape 
        (nparticles, latent_dim)

    """
    for gnn in self.gnn_stacks:
      x, edge_features = gnn(x, edge_index, edge_features, positions)
    return x, edge_features


class Decoder(nn.Module):
  """The Decoder: :math: `\mathcal{G} \rightarrow \mathcal{Y}` extracts the 
  dynamics information from the nodes of the final latent graph, 
  :math: `y_i = \delta v (v_i^M)`

  """

  def __init__(
          self,
          nnode_in: int,
          nnode_out: int,
          nmlp_layers: int,
          mlp_hidden_dim: int):
    """The Decoder coder's learned function, :math: `\detla v`, is an MLP. 
    After the Decoder, the future position and velocity are updated using an 
    Euler integrator, so the :math: `yi` corresponds to accelerations, 
    :math: `\"{p}_i`, with 2D or 3D dimension, depending on the physical domain.

    Args:
      nnode_in: Number of node inputs (latent dimension of size 128).
      nnode_out: Number of node outputs (particle dimension).
      nmlp_layer: Number of hidden layers in the MLP (typically of size 2).
      mlp_hidden_dim: Size of the hidden layer (latent dimension of size 128).
    """
    super(Decoder, self).__init__()
    self.node_fn = build_mlp(
        nnode_in, [mlp_hidden_dim for _ in range(nmlp_layers)], nnode_out)

  def forward(self,
              x: torch.tensor):
    """The forward hook runs when the Decoder class is instantiated

    Args:
      x: Particle state representation as a torch tensor with shape 
        (nparticles, nnode_in)

    """
    return self.node_fn(x)


class EncodeProcessDecode(nn.Module):
  def __init__(
      self,
      nnode_in_features: int,
      nnode_out_features: int,
      nedge_in_features: int,
      latent_dim: int,
      nmessage_passing_steps: int,
      nmlp_layers: int,
      mlp_hidden_dim: int,
  ):
    """Encode-Process-Decode function approximator for learnable simulator.

    Args:
      nnode_in_features: Number of node input features (for 2D = 30, 
        calculated as [10 = 5 times steps * 2 positions (x, y) + 
        4 distances to boundaries (top/bottom/left/right) + 
        16 particle type embeddings]).
      nnode_out_features:  Number of node outputs (particle dimension).
      nedge_in_features: Number of edge input features (for 2D = 3, 
        calculated as [2 (x, y) relative displacements between 2 particles + 
        distance between 2 particles]).
      latent_dim: Size of latent dimension (128)
      nmlp_layer: Number of hidden layers in the MLP (typically of size 2).
      mlp_hidden_dim: Size of the hidden layer (latent dimension of size 128).

    """
    super(EncodeProcessDecode, self).__init__()
    self._encoder = Encoder(
        nnode_in_features=nnode_in_features,
        nnode_out_features=latent_dim,
        nedge_in_features=nedge_in_features,
        nedge_out_features=latent_dim,
        nmlp_layers=nmlp_layers,
        mlp_hidden_dim=mlp_hidden_dim,
    )
    # self._processor = TransformerProcessor(
    #     nnode_in=latent_dim,
    #     nnode_out=latent_dim,
    #     nedge_in=latent_dim,
    #     nedge_out=latent_dim,
    #     nmessage_passing_steps=nmessage_passing_steps,
    #     nmlp_layers=nmlp_layers,
    #     mlp_hidden_dim=mlp_hidden_dim,
    # )
    self._processor = PhysicsTransformerProcessor(
        nnode_in=latent_dim,
        nnode_out=latent_dim,
        nedge_in=latent_dim,
        nedge_out=latent_dim,
        nmessage_passing_steps=nmessage_passing_steps,
        nmlp_layers=nmlp_layers,
        mlp_hidden_dim=mlp_hidden_dim,
    )

    ### For local attention with radius masking
    # self._processor = PhysicsTransformerProcessorRadiusMasked(
    #     nnode_in=latent_dim,
    #     nnode_out=latent_dim,
    #     nedge_in=latent_dim,
    #     nedge_out=latent_dim,
    #     nmessage_passing_steps=nmessage_passing_steps,
    #     nmlp_layers=nmlp_layers,
    #     mlp_hidden_dim=mlp_hidden_dim,
    #     radius=0.2,  # neighbor cutoff distance; tune as needed
    #     n_heads=8,
    #     dropout=0.1
    # )

    # self._processor = PointTransformerProcessor(
    #     latent_dim=latent_dim,
    #     pos_dim=2,  # assuming 2D positions; change if 3D
    #     n_layers=nmessage_passing_steps,
    #     n_heads=8,
    #     pos_mlp_hidden=mlp_hidden_dim,
    #     dropout=0.1
    # )

    self._decoder = Decoder(
        nnode_in=latent_dim,
        nnode_out=nnode_out_features,
        nmlp_layers=nmlp_layers,
        mlp_hidden_dim=mlp_hidden_dim,
    )

  def forward(self,
              x: torch.tensor,
              edge_index: torch.tensor,
              edge_features: torch.tensor,
              most_recent_position: torch.tensor = None):
    """The forward hook runs at instatiation of EncodeProcessorDecode class.

      Args:
        x: Particle state representation as a torch tensor with shape 
          (nparticles, nnode_in_features)
        edge_index: A torch tensor list of source and target nodes with shape 
          (2, nedges)
        edge_features: Edge features as a torch tensor with shape 
          (nedges, nedge_in_features)
          
      Returns:
        x: Particle state representation as a torch tensor with shape
          (nparticles, nnode_out_features)
    """
    x, edge_features = self._encoder(x, edge_features)
    x, edge_features = self._processor(x, edge_index, edge_features, most_recent_position)
    x = self._decoder(x)
    return x

