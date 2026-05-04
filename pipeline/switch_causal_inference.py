# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# To view a copy of this license, visit http://www.apache.org/licenses/LICENSE-2.0
#
# No warranties are given. The work is provided "AS IS", without warranty of any kind, express or implied.
#
# SPDX-License-Identifier: Apache-2.0
from typing import List, Optional
import math
import torch
import torch.nn.functional as F

from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
from utils.memory import gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation
from pipeline.causal_inference import CausalInferencePipeline
import torch.distributed as dist
from utils.debug_option import DEBUG


class SwitchCausalInferencePipeline(CausalInferencePipeline):
    def __init__(
        self,
        args,
        device,
        *,
        generator: WanDiffusionWrapper | None = None,
        text_encoder: WanTextEncoder | None = None,
        vae: WanVAEWrapper | None = None,
    ):
        super().__init__(args, device, generator=generator, text_encoder=text_encoder, vae=vae)
        self.global_sink = getattr(args, "global_sink", False)
        self.semantic_bridge_enabled = getattr(args, "semantic_bridge_enabled", True)
        self.semantic_bridge_recent_frames = max(1, int(getattr(args, "semantic_bridge_recent_frames", 2)))
        self.semantic_bridge_micro_recache_frames = max(0, int(getattr(args, "semantic_bridge_micro_recache_frames", self.num_frame_per_block)))
        self.semantic_bridge_tau_low = float(getattr(args, "semantic_bridge_tau_low", 0.15))
        self.semantic_bridge_tau_high = float(getattr(args, "semantic_bridge_tau_high", 0.45))
        self.semantic_bridge_scale = float(getattr(args, "semantic_bridge_scale", 1.0))
        self.semantic_bridge_decay = float(getattr(args, "semantic_bridge_decay", 0.85))

        self.base_local_attn_size = int(getattr(args.model_kwargs, "local_attn_size", -1))
        self.adaptive_prompt_window_enabled = bool(getattr(args, "adaptive_prompt_window_enabled", True)) and self.base_local_attn_size != -1
        self.adaptive_prompt_window_min_size = int(
            getattr(args, "adaptive_prompt_window_min_size", max(3, self.base_local_attn_size // 2 if self.base_local_attn_size != -1 else 3))
        )
        self.adaptive_prompt_window_tau_post = float(
            getattr(args, "adaptive_prompt_window_tau_post", max(1, self.base_local_attn_size if self.base_local_attn_size != -1 else self.num_frame_per_block * 2))
        )
        self.adaptive_prompt_window_tau_pre = float(
            getattr(args, "adaptive_prompt_window_tau_pre", max(1, self.num_frame_per_block * 2))
        )
        self.segment_anchor_enabled = bool(getattr(args, "segment_anchor_enabled", True)) and self.base_local_attn_size != -1
        self.segment_anchor_recent_frames = max(1, int(getattr(args, "segment_anchor_recent_frames", max(1, self.num_frame_per_block))))
        self.segment_anchor_max_segments = max(1, int(getattr(args, "segment_anchor_max_segments", 4)))
        self.segment_anchor_semantic_mix = float(getattr(args, "segment_anchor_semantic_mix", 0.35))
        self.segment_anchor_scale = float(getattr(args, "segment_anchor_scale", 0.8))

    def _clear_crossattn_cache(self):
        for blk in self.crossattn_cache:
            blk["k"].zero_()
            blk["v"].zero_()
            blk["is_init"] = False

    def _clear_semantic_bridge(self):
        for block_idx in range(self.num_transformer_blocks):
            cache = self.kv_cache1[block_idx]
            cache.pop("semantic_bridge_k", None)
            cache.pop("semantic_bridge_v", None)
            cache.pop("semantic_bridge_scale", None)

    def _clear_segment_anchors(self):
        for block_idx in range(self.num_transformer_blocks):
            cache = self.kv_cache1[block_idx]
            cache.pop("segment_anchor_k", None)
            cache.pop("segment_anchor_v", None)
            cache.pop("segment_anchor_scale", None)

    def _get_projected_prompt_signature(self, conditional_dict):
        cached = conditional_dict.get("_projected_prompt_signature")
        if cached is not None:
            return cached
        prompt_embeds = conditional_dict["prompt_embeds"]
        device = next(self.generator.parameters()).device
        with torch.no_grad():
            prompt_embeds = prompt_embeds.to(device)
            projected = self.generator.model.text_embedding(prompt_embeds)
            mask = prompt_embeds.abs().sum(dim=-1).gt(0)
            denom = mask.sum(dim=1, keepdim=True).clamp(min=1)
            signature = (projected * mask.unsqueeze(-1).to(projected.dtype)).sum(dim=1) / denom.to(projected.dtype)
        conditional_dict["_projected_prompt_signature"] = signature.detach()
        return conditional_dict["_projected_prompt_signature"]

    def _compute_switch_strength(self, old_conditional_dict, new_conditional_dict):
        old_sig = self._get_projected_prompt_signature(old_conditional_dict).float()
        new_sig = self._get_projected_prompt_signature(new_conditional_dict).float()
        cosine = F.cosine_similarity(old_sig, new_sig, dim=-1, eps=1e-6).clamp(-1.0, 1.0)
        rho = (1.0 - cosine).clamp(0.0, 1.0)
        return old_sig, new_sig, rho

    def _project_to_motion_orthogonal_subspace(self, delta_heads, v_cache):
        frame_tokens = int(self.frame_seq_length)
        valid_tokens = int(v_cache.shape[1])
        if frame_tokens <= 0 or valid_tokens < frame_tokens * 2:
            return delta_heads

        last_summary = v_cache[:, valid_tokens - frame_tokens:valid_tokens].mean(dim=1, keepdim=True)
        prev_start = valid_tokens - frame_tokens * 2
        prev_end = valid_tokens - frame_tokens
        prev_summary = v_cache[:, prev_start:prev_end].mean(dim=1, keepdim=True)
        motion_dir = (last_summary - prev_summary).to(dtype=delta_heads.dtype)

        motion_norm_sq = motion_dir.pow(2).sum(dim=-1, keepdim=True)
        proj_coeff = (delta_heads * motion_dir).sum(dim=-1, keepdim=True) / (motion_norm_sq + 1e-6)
        delta_orth = delta_heads - proj_coeff * motion_dir
        low_motion_mask = motion_norm_sq <= 1e-6
        delta_orth = torch.where(low_motion_mask, delta_heads, delta_orth)
        return delta_orth

    def _install_semantic_bridge(self, old_conditional_dict, new_conditional_dict):
        old_sig, new_sig, rho = self._compute_switch_strength(old_conditional_dict, new_conditional_dict)
        rho_max = float(rho.max().item())
        delta_sig = (new_sig - old_sig).to(dtype=self.kv_cache1[0]["v"].dtype, device=self.kv_cache1[0]["v"].device)
        batch_size = delta_sig.shape[0]
        for block_idx in range(self.num_transformer_blocks):
            cache = self.kv_cache1[block_idx]
            valid_end = int(cache["local_end_index"].item())
            if valid_end <= 0:
                cache.pop("semantic_bridge_k", None)
                cache.pop("semantic_bridge_v", None)
                cache.pop("semantic_bridge_scale", None)
                continue

            v_cache = cache["v"][:, :valid_end]
            if v_cache.shape[1] == 0:
                continue
            num_heads = v_cache.shape[2]
            head_dim = v_cache.shape[3]
            delta_heads = delta_sig.view(batch_size, num_heads, head_dim).unsqueeze(1)
            delta_heads = self._project_to_motion_orthogonal_subspace(delta_heads, v_cache)

            recent_tokens = min(valid_end, self.semantic_bridge_recent_frames * self.frame_seq_length)
            recent_summary = v_cache[:, valid_end - recent_tokens:valid_end].mean(dim=1, keepdim=True)

            sink_size = getattr(self.generator.model.blocks[block_idx].self_attn, "sink_size", 0)
            sink_tokens = min(valid_end, sink_size * self.frame_seq_length)
            if sink_tokens > 0:
                sink_summary = v_cache[:, :sink_tokens].mean(dim=1, keepdim=True)
            else:
                sink_summary = recent_summary

            recent_align = F.cosine_similarity(
                recent_summary.squeeze(1).float(),
                delta_heads.squeeze(1).float(),
                dim=-1,
                eps=1e-6,
            ).clamp(min=0.0, max=1.0)
            sink_align = F.cosine_similarity(
                sink_summary.squeeze(1).float(),
                delta_heads.squeeze(1).float(),
                dim=-1,
                eps=1e-6,
            ).clamp(min=0.0, max=1.0)
            rho_heads = rho.to(dtype=v_cache.dtype, device=v_cache.device).view(batch_size, 1)
            recent_gate = (rho_heads * recent_align.to(dtype=v_cache.dtype)).unsqueeze(1).unsqueeze(-1)
            sink_gate = (rho_heads * sink_align.to(dtype=v_cache.dtype)).unsqueeze(1).unsqueeze(-1)
            recent_bridge = (1.0 - recent_gate) * recent_summary + recent_gate * delta_heads
            sink_bridge = (1.0 - sink_gate) * sink_summary + sink_gate * delta_heads
            bridge = torch.cat([sink_bridge, recent_bridge], dim=1).contiguous()
            cache["semantic_bridge_k"] = bridge
            cache["semantic_bridge_v"] = bridge.clone()
            cache["semantic_bridge_scale"] = torch.full(
                (batch_size, 1, 1, 1),
                float(self.semantic_bridge_scale),
                device=bridge.device,
                dtype=bridge.dtype,
            )
        return rho_max

    def _decay_semantic_bridge(self):
        if not self.semantic_bridge_enabled:
            return
        for block_idx in range(self.num_transformer_blocks):
            cache = self.kv_cache1[block_idx]
            bridge_scale = cache.get("semantic_bridge_scale")
            if bridge_scale is None:
                continue
            bridge_scale.mul_(self.semantic_bridge_decay)
            if float(bridge_scale.max().item()) < 1e-3:
                cache.pop("semantic_bridge_k", None)
                cache.pop("semantic_bridge_v", None)
                cache.pop("semantic_bridge_scale", None)

    def _compute_phase_window_frames(self, current_start_frame: int, segment_start_frame: int, next_switch_pos: int | None):
        if not self.adaptive_prompt_window_enabled or self.base_local_attn_size == -1:
            return self.base_local_attn_size
        sink_size = int(getattr(self.generator.model.blocks[0].self_attn, "sink_size", 0))
        min_size = max(sink_size + 1, min(self.adaptive_prompt_window_min_size, self.base_local_attn_size))
        age = max(0, current_start_frame - segment_start_frame)
        post_weight = math.exp(-float(age) / max(self.adaptive_prompt_window_tau_post, 1e-6))
        if next_switch_pos is None:
            pre_weight = 0.0
        else:
            dist_to_next = max(0, next_switch_pos - current_start_frame)
            pre_weight = math.exp(-float(dist_to_next) / max(self.adaptive_prompt_window_tau_pre, 1e-6))
        phase_weight = max(post_weight, pre_weight)
        window = int(round(min_size + (self.base_local_attn_size - min_size) * phase_weight))
        return max(min_size, min(self.base_local_attn_size, window))

    def _set_adaptive_window_frames(self, window_frames: int):
        if self.base_local_attn_size == -1:
            return
        window_frames = max(1, min(self.base_local_attn_size, int(window_frames)))
        adaptive_window_tokens = window_frames * self.frame_seq_length
        for block_idx in range(self.num_transformer_blocks):
            self.kv_cache1[block_idx]["adaptive_window_tokens"] = adaptive_window_tokens

    def _build_segment_anchor_from_current_cache(self, conditional_dict):
        if not self.segment_anchor_enabled:
            return None
        prompt_sig = self._get_projected_prompt_signature(conditional_dict).to(
            dtype=self.kv_cache1[0]["v"].dtype,
            device=self.kv_cache1[0]["v"].device,
        )
        batch_size = prompt_sig.shape[0]
        anchors = []
        for block_idx in range(self.num_transformer_blocks):
            cache = self.kv_cache1[block_idx]
            valid_end = int(cache["local_end_index"].item())
            if valid_end <= 0:
                anchors.append(None)
                continue
            v_cache = cache["v"][:, :valid_end]
            if v_cache.shape[1] == 0:
                anchors.append(None)
                continue
            num_heads = v_cache.shape[2]
            head_dim = v_cache.shape[3]
            prompt_heads = prompt_sig.view(batch_size, num_heads, head_dim).unsqueeze(1)
            recent_tokens = min(valid_end, self.segment_anchor_recent_frames * self.frame_seq_length)
            recent_summary = v_cache[:, valid_end - recent_tokens:valid_end].mean(dim=1, keepdim=True)
            semantic_mix = self.segment_anchor_semantic_mix
            anchor = ((1.0 - semantic_mix) * recent_summary + semantic_mix * prompt_heads).contiguous()
            anchors.append(anchor)
        return anchors

    def _append_segment_anchors(self, anchors):
        if (anchors is None) or (not self.segment_anchor_enabled):
            return
        for block_idx, anchor in enumerate(anchors):
            if anchor is None:
                continue
            cache = self.kv_cache1[block_idx]
            old_k = cache.get("segment_anchor_k")
            old_v = cache.get("segment_anchor_v")
            old_s = cache.get("segment_anchor_scale")
            if old_k is None:
                new_k = anchor
                new_v = anchor.clone()
                new_s = torch.full((anchor.shape[0], 1, 1, 1), float(self.segment_anchor_scale), device=anchor.device, dtype=anchor.dtype)
            else:
                new_k = torch.cat([old_k, anchor], dim=1)
                new_v = torch.cat([old_v, anchor.clone()], dim=1)
                new_scale = torch.full((anchor.shape[0], 1, 1, 1), float(self.segment_anchor_scale), device=anchor.device, dtype=anchor.dtype)
                new_s = torch.cat([old_s, new_scale], dim=1)
                if new_k.shape[1] > self.segment_anchor_max_segments:
                    new_k = new_k[:, -self.segment_anchor_max_segments:]
                    new_v = new_v[:, -self.segment_anchor_max_segments:]
                    new_s = new_s[:, -self.segment_anchor_max_segments:]
            cache["segment_anchor_k"] = new_k
            cache["segment_anchor_v"] = new_v
            cache["segment_anchor_scale"] = new_s

    def _recache_after_switch(self, output, current_start_frame, new_conditional_dict):
        self._clear_semantic_bridge()
        if self.base_local_attn_size != -1:
            self._set_adaptive_window_frames(self.base_local_attn_size)
        if not self.global_sink:
            for block_idx in range(self.num_transformer_blocks):
                cache = self.kv_cache1[block_idx]
                cache["k"].zero_()
                cache["v"].zero_()

        self._clear_crossattn_cache()
        if current_start_frame == 0:
            return

        num_recache_frames = current_start_frame if self.local_attn_size == -1 else min(self.local_attn_size, current_start_frame)
        recache_start_frame = current_start_frame - num_recache_frames
        frames_to_recache = output[:, recache_start_frame:current_start_frame]
        if frames_to_recache.device.type == 'cpu':
            target_device = next(self.generator.parameters()).device
            frames_to_recache = frames_to_recache.to(target_device)
        batch_size = frames_to_recache.shape[0]
        print(f"[semantic-bridge] full recache frames={num_recache_frames}, start={recache_start_frame}, current={current_start_frame}")

        device = frames_to_recache.device
        block_mask = self.generator.model._prepare_blockwise_causal_attn_mask(
            device=device,
            num_frames=num_recache_frames,
            frame_seqlen=self.frame_seq_length,
            num_frame_per_block=self.num_frame_per_block,
            local_attn_size=self.local_attn_size,
        )
        context_timestep = torch.ones([batch_size, num_recache_frames], device=device, dtype=torch.int64) * self.args.context_noise
        self.generator.model.block_mask = block_mask

        with torch.no_grad():
            self.generator(
                noisy_image_or_video=frames_to_recache,
                conditional_dict=new_conditional_dict,
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=recache_start_frame * self.frame_seq_length,
                sink_recache_after_switch=not self.global_sink,
            )
        self._clear_crossattn_cache()

    def _micro_recache_after_switch(self, output, current_start_frame, new_conditional_dict, micro_frames):
        if self.base_local_attn_size != -1:
            self._set_adaptive_window_frames(self.base_local_attn_size)
        self._clear_crossattn_cache()
        if current_start_frame == 0 or micro_frames <= 0:
            return
        num_recache_frames = min(micro_frames, current_start_frame)
        recache_start_frame = current_start_frame - num_recache_frames
        frames_to_recache = output[:, recache_start_frame:current_start_frame]
        if frames_to_recache.device.type == 'cpu':
            target_device = next(self.generator.parameters()).device
            frames_to_recache = frames_to_recache.to(target_device)
        batch_size = frames_to_recache.shape[0]
        print(f"[semantic-bridge] micro recache frames={num_recache_frames}, start={recache_start_frame}, current={current_start_frame}")

        device = frames_to_recache.device
        block_mask = self.generator.model._prepare_blockwise_causal_attn_mask(
            device=device,
            num_frames=num_recache_frames,
            frame_seqlen=self.frame_seq_length,
            num_frame_per_block=self.num_frame_per_block,
            local_attn_size=self.local_attn_size,
        )
        context_timestep = torch.ones([batch_size, num_recache_frames], device=device, dtype=torch.int64) * self.args.context_noise
        self.generator.model.block_mask = block_mask

        with torch.no_grad():
            self.generator(
                noisy_image_or_video=frames_to_recache,
                conditional_dict=new_conditional_dict,
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=recache_start_frame * self.frame_seq_length,
                sink_recache_after_switch=False,
            )
        self._clear_crossattn_cache()

    def _apply_switch_memory_policy(self, output, current_start_frame, old_conditional_dict, new_conditional_dict):
        if not self.semantic_bridge_enabled:
            self._recache_after_switch(output, current_start_frame, new_conditional_dict)
            return "full-recache", 1.0

        _, _, rho = self._compute_switch_strength(old_conditional_dict, new_conditional_dict)
        rho_max = float(rho.max().item())
        if rho_max >= self.semantic_bridge_tau_high:
            self._recache_after_switch(output, current_start_frame, new_conditional_dict)
            return "full-recache", rho_max

        self._install_semantic_bridge(old_conditional_dict, new_conditional_dict)
        self._clear_crossattn_cache()
        if rho_max >= self.semantic_bridge_tau_low:
            self._micro_recache_after_switch(
                output,
                current_start_frame,
                new_conditional_dict,
                micro_frames=self.semantic_bridge_micro_recache_frames,
            )
            return "bridge+micro-recache", rho_max
        return "bridge-only", rho_max

    def inference(
        self,
        noise: torch.Tensor,
        *,
        text_prompts_first: List[str],
        text_prompts_second: List[str],
        switch_frame_index: int,
        initial_latent: Optional[torch.Tensor] = None,
        return_latents: bool = False,
        low_memory: bool = False,
    ):
        batch_size, num_output_frames, num_channels, height, width = noise.shape
        assert num_output_frames % self.num_frame_per_block == 0
        num_blocks = num_output_frames // self.num_frame_per_block

        cond_first = self.text_encoder(text_prompts=text_prompts_first)
        cond_second = self.text_encoder(text_prompts=text_prompts_second)
        self._get_projected_prompt_signature(cond_first)
        self._get_projected_prompt_signature(cond_second)

        if low_memory:
            gpu_memory_preservation = get_cuda_free_memory_gb(gpu) + 5
            move_model_to_device_with_memory_preservation(self.text_encoder, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)

        output_device = torch.device('cpu') if low_memory else noise.device
        output = torch.zeros([batch_size, num_output_frames, num_channels, height, width], device=output_device, dtype=noise.dtype)

        local_attn_cfg = getattr(self.args.model_kwargs, "local_attn_size", -1)
        if local_attn_cfg != -1:
            kv_cache_size = local_attn_cfg * self.frame_seq_length
            kv_policy = f"int->local, size={local_attn_cfg}"
        else:
            kv_cache_size = num_output_frames * self.frame_seq_length
            kv_policy = "global (-1)"
        print(f"kv_cache_size: {kv_cache_size} (policy: {kv_policy}, frame_seq_length: {self.frame_seq_length}, num_output_frames: {num_output_frames})")

        self._initialize_kv_cache(batch_size, dtype=noise.dtype, device=noise.device, kv_cache_size_override=kv_cache_size)
        self._initialize_crossattn_cache(batch_size=batch_size, dtype=noise.dtype, device=noise.device)
        self._clear_semantic_bridge()
        self._clear_segment_anchors()

        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        current_start_frame = 0
        segment_start_frame = 0
        next_switch_pos = switch_frame_index
        self.generator.model.local_attn_size = self.local_attn_size
        print(f"[inference] local_attn_size set on model: {self.generator.model.local_attn_size}")
        self._set_all_modules_max_attention_size(self.local_attn_size)

        all_num_frames = [self.num_frame_per_block] * num_blocks
        using_second = False
        last_window_frames = None
        for current_num_frames in all_num_frames:
            if (not using_second) and (current_start_frame >= switch_frame_index):
                completed_segment_anchor = self._build_segment_anchor_from_current_cache(cond_first)
                switch_mode, rho_max = self._apply_switch_memory_policy(output, current_start_frame, cond_first, cond_second)
                self._append_segment_anchors(completed_segment_anchor)
                using_second = True
                segment_start_frame = current_start_frame
                next_switch_pos = None
                print(f"[semantic-bridge] switch_frame_index={switch_frame_index}, current_start_frame={current_start_frame}, mode={switch_mode}, rho_max={rho_max:.4f}")

            active_window_frames = self._compute_phase_window_frames(current_start_frame, segment_start_frame, next_switch_pos)
            if active_window_frames != -1:
                self._set_adaptive_window_frames(active_window_frames)
                if active_window_frames != last_window_frames:
                    seg_name = 1 if using_second else 0
                    print(f"[adaptive-window] current_start_frame={current_start_frame}, segment_idx={seg_name}, active_window_frames={active_window_frames}")
                    last_window_frames = active_window_frames

            cond_in_use = cond_second if using_second else cond_first
            noisy_input = noise[:, current_start_frame - num_input_frames: current_start_frame + current_num_frames - num_input_frames]
            for index, current_timestep in enumerate(self.denoising_step_list):
                timestep = torch.ones([batch_size, current_num_frames], device=noise.device, dtype=torch.int64) * current_timestep
                if index < len(self.denoising_step_list) - 1:
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=cond_in_use,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                    )
                    next_timestep = self.denoising_step_list[index + 1]
                    noisy_input = self.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        next_timestep * torch.ones([batch_size * current_num_frames], device=noise.device, dtype=torch.long),
                    ).unflatten(0, denoised_pred.shape[:2])
                else:
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=cond_in_use,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                    )

            output[:, current_start_frame: current_start_frame + current_num_frames] = denoised_pred.to(output.device)
            context_timestep = torch.ones_like(timestep) * self.args.context_noise
            self.generator(
                noisy_image_or_video=denoised_pred,
                conditional_dict=cond_in_use,
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
            )
            self._decay_semantic_bridge()
            current_start_frame += current_num_frames

        video = self.vae.decode_to_pixel(output.to(noise.device), use_cache=False)
        video = (video * 0.5 + 0.5).clamp(0, 1)
        if return_latents:
            return video, output
        return video
