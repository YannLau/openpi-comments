import dataclasses


import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.models.model import Observation


#### DummyModelConfig — 继承 BaseModelConfig
@dataclasses.dataclass(frozen=True)
class DummyModelConfig(_model.BaseModelConfig):
    """与 Pi0Config 相同的输入/输出规范，但背后只创建一个简单 MLP。"""

    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = 48
    hidden_dim: int = 256  # MLP 隐藏层大小
    num_layers: int = 3  # MLP 层数
    embed_dim: int = 64  # 文本 token 嵌入维度

    @property
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.DUMMY

    def create(self, rng) -> "DummyModel":
        return DummyModel(self, rngs=nnx.Rngs(rng))

    def inputs_spec(self, *, batch_size=1):
        """与 Pi0Config.inputs_spec() 完全相同！"""
        image_spec = jax.ShapeDtypeStruct([batch_size, 224, 224, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)
        with at.disable_typechecking():
            obs = Observation(
                images={"base_0_rgb": image_spec, "left_wrist_0_rgb": image_spec, "right_wrist_0_rgb": image_spec},
                image_masks={k: image_mask_spec for k in ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]},
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        act = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)
        return obs, act


#### DummyModel — 继承 BaseModel（nnx.Module）
class DummyModel(_model.BaseModel):
    def __init__(self, config: DummyModelConfig, rngs: nnx.Rngs):
        super().__init__(
            action_dim=config.action_dim, action_horizon=config.action_horizon, max_token_len=config.max_token_len
        )

        # 文本 token 嵌入层（将 token IDs 映射为连续向量）
        self.embedding = nnx.Embed(num_embeddings=257152, features=config.embed_dim, rngs=rngs)

        # 计算输入维度：
        #   3 个图像 × 3 通道（全局平均池化后） + 状态维度 + 文本嵌入平均
        input_dim = 3 * 3 + config.action_dim + config.embed_dim

        # MLP 隐藏层
        layers = []
        dims = [input_dim] + [config.hidden_dim] * config.num_layers
        for i in range(len(dims) - 1):
            layers.append(nnx.Linear(dims[i], dims[i + 1], rngs=rngs))
        self.hidden_layers = layers

        # 输出层：预测 (action_horizon × action_dim) 个值
        self.output_layer = nnx.Linear(dims[-1], config.action_horizon * config.action_dim, rngs=rngs)

    def _forward(self, obs: Observation) -> jax.Array:
        """将 Observation 映射为扁平的动作预测。"""
        batch_size = obs.state.shape[0]

        # 1) 处理图像：全局平均池化 (B, H, W, C) → (B, C)
        image_features = []
        for key in ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]:
            img = obs.images[key]  # (B, 224, 224, 3)
            pooled = jnp.mean(img, axis=(1, 2))  # (B, 3)
            image_features.append(pooled)
        image_vec = jnp.concatenate(image_features, axis=-1)  # (B, 9)

        # 2) 状态
        state_vec = obs.state  # (B, action_dim)

        # 3) 文本 prompt：embedding → 平均
        prompt_ids = jnp.clip(obs.tokenized_prompt, 0, 257151)  # (B, max_token_len)
        prompt_embeds = self.embedding(prompt_ids)  # (B, max_token_len, embed_dim)
        prompt_mask = obs.tokenized_prompt_mask[..., None]  # (B, max_token_len, 1)
        prompt_vec = jnp.sum(prompt_embeds * prompt_mask, axis=1) / (jnp.sum(prompt_mask, axis=1) + 1e-8)
        # → (B, embed_dim)

        # 4) 拼接
        x = jnp.concatenate([image_vec, state_vec, prompt_vec], axis=-1)  # (B, 9 + ad + ed)

        # 5) MLP
        for layer in self.hidden_layers:
            x = nnx.relu(layer(x))

        # 6) 输出 → reshape 为 (B, action_horizon, action_dim)
        flat_actions = self.output_layer(x)
        return flat_actions.reshape(batch_size, self.action_horizon, self.action_dim)

    def compute_loss(self, rng, observation, actions, *, train=False):
        pred_actions = self._forward(observation)
        # 逐时间步 MSE 损失（与 pi0 保持一致）
        return jnp.mean((pred_actions - actions) ** 2, axis=-1)  # (B, action_horizon)

    def sample_actions(self, rng, observation, **kwargs):
        return self._forward(observation)
