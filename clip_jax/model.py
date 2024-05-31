import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from haiku import LayerNorm

def create_bn(name):
    init = hk.initializers.RandomUniform()
    return hk.BatchNorm(True, True, 0.1, scale_init=init, offset_init=init, name=name, data_format="NCHW")

class PositionalEmbedding(hk.Module):
    def __init__(self, spacial_dim, embed_dim):
        super().__init__(name="pos_e")
        self.positional_embedding = hk.get_parameter("pos_embd", (spacial_dim ** 2 + 1, embed_dim), init=hk.initializers.RandomNormal())
    def __call__(self, x):
        return x + self.positional_embedding[:, None, :]

class AttentionPool2d(hk.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__(name="attnpool")
        self.pe = PositionalEmbedding(spacial_dim, embed_dim)
        self.num_heads = num_heads
        self.mha = hk.MultiHeadAttention(num_heads, 
        embed_dim // num_heads, 
        w_init_scale=0.1, 
        value_size=None, 
        model_size=output_dim or embed_dim, 
        name="mha")

    def __call__(self, x):
        N, C, H, W = x.shape
        x = x.reshape(N, C, H * W).transpose(2, 0, 1)  # NCHW -> (HW)NC
        n = x.mean(axis=0, keepdims=True)
        x = jax.numpy.concatenate([n, x], axis=0)  # (HW+1)NC
        x = self.pe(x).transpose(1, 0, 2)
        x = self.mha(x[:1], x, x)
        return x

class Bottleneck(hk.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, layer_num=0, num=0):
        super().__init__(name=f'layer_{layer_num}_{num}')

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = hk.Conv2D(planes, kernel_shape=(1, 1), stride=1, padding="SAME", with_bias=False, name="conv1", data_format="NCHW")
        self.bn1 = create_bn(name="bn1")
        self.conv2 = hk.Conv2D(planes, kernel_shape=(3, 3), stride=1, padding="SAME", with_bias=False, name="conv2", data_format="NCHW")
        self.bn2 = create_bn(name="bn2")
        self.avgpool = hk.AvgPool(stride, stride, padding="SAME", channel_axis=1) if stride > 1 else None
        self.conv3 = hk.Conv2D(planes * self.expansion, kernel_shape=(1, 1), stride=1, padding="SAME", with_bias=False, name="conv3", data_format="NCHW")
        self.bn3 = create_bn(name="bn3")
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = hk.Sequential([
                hk.AvgPool(stride, stride, padding="SAME", channel_axis=1),
                hk.Conv2D(planes * self.expansion, kernel_shape=(1, 1), stride=1, with_bias=False, name="downsample_conv", data_format="NCHW")
            ]
            )
            self.downsample_bn = hk.BatchNorm(True, True, 0.1, name="downsample_bn", data_format="NCHW")
        

    def __call__(self, x):
        identity = x
        out = jax.nn.relu(self.bn1(self.conv1(x), True))
        out = jax.nn.relu(self.bn2(self.conv2(out), True))
        if self.stride > 1:
            out = self.avgpool(out)
        out = self.bn3(self.conv3(out), True)

        if self.downsample is not None:
            identity = self.downsample(x)
            identity = self.downsample_bn(identity, True)
        
        out += identity
        out = jax.nn.relu(out)
        return out


@hk.transparent
class ModifiedResNet(hk.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64, name=None):
        super().__init__(name="visual")
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = hk.Conv2D(width // 2, 
            kernel_shape=(3, 3), 
            stride=(2, 2), 
            padding="SAME", 
            with_bias=False, 
            name="conv1", data_format="NCHW")
        self.bn1 = create_bn(name="bn1")

        self.conv2 = hk.Conv2D(width // 2, 
            kernel_shape=(3, 3), 
            stride=(1, 1), 
            padding="SAME", 
            with_bias=False,
            name="conv2", data_format="NCHW")
        self.bn2 = create_bn(name="bn2")

        self.conv3 = hk.Conv2D(width,
            kernel_shape=(3, 3), 
            stride=(1, 1), 
            padding="SAME",
            with_bias=False,
            name="conv3", data_format="NCHW")
        self.bn3 = create_bn(name="bn3")

        self.avgpool = hk.AvgPool(2, 2, padding="SAME", channel_axis=1)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0], layer_num=1)
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2, layer_num=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2, layer_num=3)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2, layer_num=4)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1, layer_num=0):
        layers = [Bottleneck(self._inplanes, planes, stride, layer_num=layer_num, num=0)]

        self._inplanes = planes * Bottleneck.expansion
        for i in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes, layer_num=layer_num, num=i))

        return hk.Sequential(layers)

    def __call__(self, x):
        def stem(x):
            x = jax.nn.relu(self.bn1(self.conv1(x), True))
            x = jax.nn.relu(self.bn2(self.conv2(x), True))
            x = jax.nn.relu(self.bn3(self.conv3(x), True))
            x = self.avgpool(x)
            return x

        #x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)
        return x

class MultiHeadAttention(hk.Module):
    def __init__(
            self,
            num_heads: int,
            head_size: int,
            w_init_scale: float,
            attn_mask: jnp.ndarray = None,
            name: str = "mha",
    ):
        super().__init__(name=name)
        self.num_heads = num_heads
        self.model_size = head_size * num_heads
        self.w_init = hk.initializers.VarianceScaling(w_init_scale)
        self.attn_mask = attn_mask

        self.in_proj_weight = hk.get_parameter("in_proj_weight", shape=[self.model_size * 3, self.model_size], init=self.w_init)
        self.in_proj_bias = hk.get_parameter("in_proj_bias", shape=[self.model_size * 3], init=self.w_init)
        self.out_proj = hk.Linear(self.model_size, name="out_proj")

    def __call__(
            self,
            x: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute (optionally masked) MHA with queries, keys & values."""
        all_out = jnp.dot(x, self.in_proj_weight.transpose())
        all_out += self.in_proj_bias

        q, k, v = jnp.array_split(all_out, 3, axis=-1)

        query_heads = self._split(q)
        key_heads = self._split(k)
        value_heads = self._split(v)

        attention_logits = jnp.einsum("tbhd,Tbhd->bhtT", query_heads, key_heads)
        sqrt_key_size = np.sqrt(self.model_size//self.num_heads).astype(k.dtype)
        attention_logits = attention_logits / sqrt_key_size

        if self.attn_mask is not None:
            attention_logits += self.attn_mask

        attention_weights = jax.nn.softmax(attention_logits)
        attention = jnp.einsum("bhtT,Tbhd->tbhd", attention_weights, value_heads)
        # Concatenate attention matrix of all heads into a single vector.
        attention_vec = jnp.reshape(attention, (*q.shape[:2], -1))

        return self.out_proj(attention_vec)

    def _split(
            self,
            x: jnp.ndarray,
    ) -> jnp.ndarray:
        return x.reshape((*x.shape[:2], self.num_heads, self.model_size//self.num_heads))


class QuickGELU(hk.Module):
    def __call__(self, x: jnp.ndarray):
        return x * jax.nn.sigmoid(1.702 * x)


class ResidualAttentionBlock(hk.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: jnp.ndarray, name: str):
        super().__init__(name=name)
        self.attn = MultiHeadAttention(n_head, d_model // n_head, 1, attn_mask, name="attn")
        self.ln_1 = LayerNorm(-1, create_scale=True, create_offset=True, name="ln_1")
        with hk.experimental.name_scope("mlp"):
            self.mlp = [hk.Linear(d_model * 4, name="c_fc"),
                        QuickGELU(),
                        hk.Linear(d_model, name="c_proj")]

        self.ln_2 = LayerNorm(-1, create_scale=True, create_offset=True, name="ln_2")

    def run_mlp(self, x: jnp.ndarray):
        for f in self.mlp:
            x = f(x)
        return x

    def __call__(self, x: jnp.ndarray):
        x = x + self.attn(self.ln_1(x))
        x = x + self.run_mlp(self.ln_2(x))
        return x


class Transformer(hk.Module):
    def __init__(self, width: int, layers: int, heads: int, name: str, attn_mask=None):
        super().__init__(name=name)
        self.width = width
        self.layers = layers
        self.resblocks = [ResidualAttentionBlock(width, heads, attn_mask, name=f"resblocks{i}") for i in range(layers)]
        self.attn_mask = attn_mask

    def __call__(self, x: jnp.ndarray):
        for b in self.resblocks:
            x = b(x)
        return x


class VisualTransformer(hk.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int,
                 name: str):
        super().__init__(name=name)

        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = hk.Conv2D(output_channels=width, kernel_shape=patch_size, stride=patch_size, with_bias=False,
                               data_format="NCHW", name="conv1")

        w_init = hk.initializers.TruncatedNormal(1. / np.sqrt(width))
        self.class_embedding = hk.get_parameter("class_embedding", shape=[width], init=w_init)
        self.positional_embedding = hk.get_parameter("positional_embedding",
                                                     shape=[(input_resolution // patch_size) ** 2 + 1, width],
                                                     init=w_init)
        self.ln_pre = LayerNorm(-1, create_scale=True, create_offset=True, name="ln_pre")

        self.transformer = Transformer(width, layers, heads, "transformer")

        self.ln_post = LayerNorm(-1, create_scale=True, create_offset=True, name="ln_post")
        self.proj = hk.get_parameter("proj", shape=[width, output_dim], init=w_init)

    def __call__(self, x: jnp.ndarray):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.transpose((0, 2, 1))  # shape = [*, grid ** 2, width]
        x = jnp.concatenate([self.class_embedding + jnp.zeros((x.shape[0], 1, x.shape[-1]), dtype=x.dtype), x],
                            axis=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding

        x = self.ln_pre(x)
        x = x.transpose((1, 0, 2))  # NLD -> LND

        x = self.transformer(x)
        x = x.transpose((1, 0, 2))  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x


class CLIP(hk.Module):
    @hk.transparent
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: int,
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        super().__init__()

        self.context_length = context_length

        vision_heads = vision_width // 64

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisualTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim,
                name="visual"
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
            name="transformer"
        )

        self.vocab_size = vocab_size
        self.token_embedding = hk.Embed(vocab_size, transformer_width, name="token_embedding")

        scale = transformer_width ** -0.5
        w_init = hk.initializers.TruncatedNormal(1. / np.sqrt(scale))
        self.positional_embedding = hk.get_parameter("positional_embedding", shape=[self.context_length, transformer_width], init=w_init)
        self.ln_final = LayerNorm(-1, create_scale=True, create_offset=True, name="ln_final")

        self.text_projection = hk.get_parameter("text_projection", shape=[transformer_width, embed_dim], init=w_init)
        self.logit_scale = hk.get_parameter("logit_scale", shape=[], init=hk.initializers.Constant(1))

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = jnp.zeros((self.context_length, self.context_length))
        mask -= 10e10
        mask = jnp.triu(mask, 1)  # zero out the lower diagonal
        return mask

    def encode_image(self, image):
        return self.visual(image)

    def encode_text(self, text):
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding
        x = x.transpose((1, 0, 2))  # NLD -> LND

        x = self.transformer(x)
        x = x.transpose((1, 0, 2))  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[jnp.arange(x.shape[0]), text.argmax(axis=-1)] @ self.text_projection

        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / jnp.linalg.norm(image_features, axis=-1, keepdims=True)
        text_features = text_features / jnp.linalg.norm(text_features, axis=-1, keepdims=True)

        # cosine similarity as logits
        logit_scale = jnp.exp(self.logit_scale)
        logits_per_image = logit_scale * image_features @ text_features.transpose()
        logits_per_text = logit_scale * text_features @ image_features.transpose()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


# get params from state dict
def get_params(state_dict: dict):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    return {
        "embed_dim": embed_dim,
        "image_resolution": image_resolution,
        "vision_layers": vision_layers,
        "vision_width": vision_width,
        "vision_patch_size": vision_patch_size,
        "context_length": context_length,
        "vocab_size": vocab_size,
        "transformer_width": transformer_width,
        "transformer_heads": transformer_heads,
        "transformer_layers": transformer_layers,
    }
