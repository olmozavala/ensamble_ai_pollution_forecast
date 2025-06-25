# %%
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
import math

class PatchEmbedding(nn.Module):
    """Creates patches from input tensor and projects them to embedding dimension."""
    def __init__(self, patch_size: tuple, in_channels: int, embedding_dim: int):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Linear(patch_size[0] * patch_size[1], embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        # Input shape: (batch, time, lat, lon)
        # Output shape: (batch, time, embedding_dim)
        patches = x.unfold(2, self.patch_size[0], self.patch_size[0]) \
                  .unfold(3, self.patch_size[1], self.patch_size[1])
        patches = patches.contiguous().view(x.size(0), x.size(1), -1)
        return self.norm(self.projection(patches))

class TransformerEncoder(nn.Module):
    """Standard Transformer encoder."""
    def __init__(self, embedding_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embedding_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.BatchNorm1d(4 * embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * embedding_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Self-attention
        attended = self.attention(x, x, x)[0]
        x = self.norm1(x + attended)
        
        # Feed forward
        # Reshape for batch norm
        orig_shape = x.shape
        x_reshaped = x.view(-1, x.size(-1))
        ff_output = self.feed_forward(x_reshaped)
        ff_output = ff_output.view(orig_shape)
        return self.norm2(x + ff_output)

class MultiStreamTransformerModel(BaseModel):
    def __init__(self, 
                 weather_time_dims: int,
                 pollution_time_dims: int,
                 weather_fields: int,
                 input_features: int,
                 weather_embedding_size: int,
                 pollution_embedding_size: int,
                 attention_heads: int,
                 lat_size: int,
                 lon_size: int,
                 dropout: float = 0.1,
                 weather_transformer_blocks: int = 2,
                 pollution_transformer_blocks: int = 2):
        super().__init__()

        # ======================== Getting the output size ========================
        # Get the number of pollutants to predict
        self.output_size = input_features - 12 # 12 is the number of time related features

        print("Buidling MultiStreamTransformerModel...")
        print(f"weather_time_dims: {weather_time_dims}")
        print(f"pollution_time_dims: {pollution_time_dims}")
        print(f"weather_fields: {weather_fields}")
        print(f"input_features: {input_features}")
        print(f"weather_embedding_size: {weather_embedding_size}")
        print(f"pollution_embedding_size: {pollution_embedding_size}")
        print(f"attention_heads: {attention_heads}")
        print(f"output_size: {self.output_size}")
        print(f"weather_transformer_blocks: {weather_transformer_blocks}")
        print(f"pollution_transformer_blocks: {pollution_transformer_blocks}")

        self.weather_time_dims = weather_time_dims
        self.pollution_time_dims = pollution_time_dims
        self.weather_fields = weather_fields
        # Weather branch - one transformer per field
        self.weather_patch_embeddings = nn.ModuleList([
            PatchEmbedding((lat_size, lon_size), 1, weather_embedding_size)
            for _ in range(weather_fields)
        ])
        
        # Create multiple transformer blocks for each weather field
        self.weather_transformers = nn.ModuleList([
            nn.ModuleList([
                TransformerEncoder(weather_embedding_size, attention_heads, dropout)
                for _ in range(weather_transformer_blocks)
            ])
            for _ in range(weather_fields)
        ])
        
        # Pollution branch
        self.pollution_embedding = nn.Linear(input_features, pollution_embedding_size)
        self.pollution_pos_encoding = nn.Parameter(
            torch.randn(pollution_time_dims, pollution_embedding_size)
        )
        # Create multiple transformer blocks for pollution
        self.pollution_transformers = nn.ModuleList([
            TransformerEncoder(pollution_embedding_size, attention_heads, dropout)
            for _ in range(pollution_transformer_blocks)
        ])
        
        # Merge layers
        # Calculate sizes based on the model configuration
        weather_merged_size = weather_embedding_size * weather_time_dims * weather_fields
        pollution_merged_size = pollution_embedding_size * pollution_time_dims
        merged_size = weather_merged_size + pollution_merged_size
        
        print(f"Initializing decoder with input size: {merged_size}")
        
        # Decoder (feed forward networks)
        self.decoder = nn.Sequential(
            nn.Linear(merged_size, merged_size // 4),  # Reduced first layer size
            nn.BatchNorm1d(merged_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(merged_size // 4, merged_size // 8),
            nn.BatchNorm1d(merged_size // 8),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(merged_size // 8, self.output_size)
        )

    def forward(self, weather_data, pollution_data):
        # weather_data shape: (batch, time, fields, lat, lon)
        # pollution_data shape: (batch, time, features)
        debug = False
        if debug:
            output_folder = "/home/olmozavala/DATA/AirPollution/TrainingData/imgs"
            batch_idx = 0
            # We will plot the weather data
            import matplotlib.pyplot as plt
            
            # Create a figure with time steps as rows and fields as columns
            num_timesteps = weather_data.shape[1]
            fig, axes = plt.subplots(num_timesteps, self.weather_fields, 
                                   figsize=(3*self.weather_fields, 3*num_timesteps))
            
            # Plot each timestep and field
            for t in range(num_timesteps):
                for field_idx in range(self.weather_fields):
                    ax = axes[t, field_idx]
                    im = ax.imshow(weather_data[batch_idx, t, field_idx, :, :].cpu().detach().numpy())
                    if t == 0:  # Only show field titles on first row
                        ax.set_title(f'Field {field_idx}')
                    if field_idx == 0:  # Only show timestep on first column
                        ax.set_ylabel(f'Time {t}')
                    plt.colorbar(im, ax=ax)
            
            plt.tight_layout()
            # Save the plot to a file
            plt.savefig(f'{output_folder}/weather_data_{batch_idx}.png')
            plt.close()
        
        # Process each weather field separately
        weather_outputs = []
        for field in range(self.weather_fields):
            # Extract single field data
            field_data = weather_data[:, :, field, :, :]  # (batch, time, lat, lon)
            
            # Create patches and embed
            embedded = self.weather_patch_embeddings[field](field_data)
            
            # Apply chained transformers
            transformed = embedded
            for transformer in self.weather_transformers[field]:
                transformed = transformer(transformed)
            
            # Concatenate across time dimension instead of pooling
            # Reshape to combine batch and time dimensions
            batch_size = transformed.size(0)
            time_steps = transformed.size(1)
            embedding_size = transformed.size(2)
            concatenated = transformed.reshape(batch_size, time_steps * embedding_size)  # (batch, time*embedding_size)
            
            weather_outputs.append(concatenated)
        
        # Combine all weather field outputs
        weather_combined = torch.cat(weather_outputs, dim=1)
        
        # Process pollution data
        pollution_embedded = self.pollution_embedding(pollution_data)
        pollution_embedded = pollution_embedded + self.pollution_pos_encoding
        
        # Apply chained transformers
        transformed = pollution_embedded
        for transformer in self.pollution_transformers:
            transformed = transformer(transformed)
        
        # Concatenate pollution time dimensions
        batch_size = transformed.size(0)
        time_steps = transformed.size(1)
        embedding_size = transformed.size(2)
        pollution_concatenated = transformed.reshape(batch_size, time_steps * embedding_size)
        
        # Merge weather and pollution branches
        merged = torch.cat([weather_combined, pollution_concatenated], dim=1)
        
        # Debug prints
        # print(f"Weather combined shape: {weather_combined.shape}")
        # print(f"Pollution concatenated shape: {pollution_concatenated.shape}")
        # print(f"Merged shape: {merged.shape}")
        
        # Decode to final output
        output = self.decoder(merged)
        
        return output

if __name__ == "__main__":
    import torch
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Define model parameters
    batch_size = 8
    weather_time_steps = 4
    weather_fields = 8
    lat_size = 25
    lon_size = 25
    pollution_time_steps = 24
    input_features = 46
    patch_size = 5    # 5x5 patches
    embedding_dim = 64
    num_heads = 4
    num_layers = 2
    dropout = 0.1
    
    # Create synthetic input data
    weather_data = torch.randn(batch_size, weather_time_steps, weather_fields, lat_size, lon_size)
    pollution_data = torch.randn(batch_size, pollution_time_steps, input_features)
    
    # %% Initialize model
    model = MultiStreamTransformerModel(
        weather_time_dims=weather_time_steps,
        pollution_time_dims=pollution_time_steps,
        weather_fields=weather_fields,
        input_features=input_features,
        weather_embedding_size=embedding_dim,
        pollution_embedding_size=embedding_dim,
        attention_heads=num_heads,
        lat_size=lat_size,
        lon_size=lon_size,
        dropout=dropout
    )
    
    # Forward pass
    print("\nInput shapes:")
    print(f"Weather data: {weather_data.shape}")
    print(f"Pollution data: {pollution_data.shape}")
    
    output = model(weather_data, pollution_data)

    # Initialize tensorboard writer
    writer = SummaryWriter('runs/model_architecture')

    # Add graph to tensorboard
    writer.add_graph(model, (weather_data, pollution_data))
    writer.close()

    print("\nOutput shape:", output.shape)
    
    # Calculate number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal number of parameters: {total_params:,}")
    
    # Test if gradients flow
    loss = output.sum()
    loss.backward()
    print("\nBackward pass completed successfully")

