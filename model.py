import torch
from torch import nn
import clip
from transformers import GPT2Model

class VQAModel(nn.Module):
    def __init__(
        self,
        vocab_size=3600,
        question_max_len=16,
        answer_max_len=10,
        hidden_size=256,
        num_layers=1,
        dropout=0.3,
        device='cpu',
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        unk_token_id=3
    ):
        super().__init__()
        self.device = device
        self.question_max_len = question_max_len
        self.answer_max_len = answer_max_len
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.fine_tuning_mode = False  # default: feature extraction

        # Token IDs
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.unk_token_id = unk_token_id

        # CLIP model (frozen by default)
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)
        for p in self.clip_model.parameters():
            p.requires_grad = False

        # GPT-2 model (frozen by default)
        self.gpt2_model = GPT2Model.from_pretrained("distilgpt2")
        self.gpt2_model.to(device)
        for p in self.gpt2_model.parameters():
            p.requires_grad = False

        # Projection layers
        self.img_proj = nn.Linear(512, hidden_size)
        self.q_proj = nn.Linear(768, hidden_size)

        # Fusion module
        self.gate_layer = nn.Linear(hidden_size*2, hidden_size)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size*3, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size)
        )

        # GRU decoder
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.gru = nn.GRU(
            input_size=hidden_size*2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.ln_gru = nn.LayerNorm(hidden_size)

        self.output = nn.Linear(hidden_size, vocab_size)
        

    # ---------------- Feature / Fine-tuning Switching ----------------
    def unfreeze_clip_layers(self, num_layers=2):
        """Unfreeze last N CLIP layers for fine-tuning"""
        self.clip_model.train()
        total_blocks = len(self.clip_model.visual.transformer.resblocks)
        for i, block in enumerate(self.clip_model.visual.transformer.resblocks):
            if i >= total_blocks - num_layers:
                for p in block.parameters():
                    p.requires_grad = True
                    p.data = p.data.float()
        # Projection and layernorm
        if hasattr(self.clip_model.visual, 'proj'):
            if isinstance(self.clip_model.visual.proj, torch.nn.Parameter):
                self.clip_model.visual.proj.requires_grad = True
                self.clip_model.visual.proj.data = self.clip_model.visual.proj.data.float()
            else:
                for p in self.clip_model.visual.proj.parameters():
                    p.requires_grad = True
                    p.data = p.data.float()
        if hasattr(self.clip_model.visual, 'ln_post'):
            for p in self.clip_model.visual.ln_post.parameters():
                p.requires_grad = True
                p.data = p.data.float()
        self.fine_tuning_mode = True
        print(f"Unfrozen last {num_layers} CLIP layers")

    def unfreeze_gpt2_layers(self, num_layers=1):
        """Unfreeze last N GPT-2 layers for fine-tuning"""
        self.gpt2_model.train()
        total_layers = len(self.gpt2_model.h)
        for i, layer in enumerate(self.gpt2_model.h):
            if i >= total_layers - num_layers:
                for p in layer.parameters():
                    p.requires_grad = True
                    p.data = p.data.float()
        # Final layer norm
        for p in self.gpt2_model.ln_f.parameters():
            p.requires_grad = True
            p.data = p.data.float()
        self.fine_tuning_mode = True
        print(f"Unfrozen last {num_layers} GPT-2 layers")

    # ---------------- Encoder functions ----------------
    def encode_image(self, images):
        """Always outputs FP32, respects fine_tuning_mode"""
        if self.fine_tuning_mode:
            img_features = self.clip_model.encode_image(images)
        else:
            with torch.no_grad():
                img_features = self.clip_model.encode_image(images)
        img_features = img_features / img_features.norm(dim=-1, keepdim=True)
        return img_features.float()

    def encode_question(self, input_ids, attention_mask):
        """Always outputs FP32, respects fine_tuning_mode"""
        if self.fine_tuning_mode:
            outputs = self.gpt2_model(input_ids=input_ids, attention_mask=attention_mask)
        else:
            with torch.no_grad():
                outputs = self.gpt2_model(input_ids=input_ids, attention_mask=attention_mask)
        
        last_hidden = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
        masked = last_hidden * mask
        sum_hidden = masked.sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1e-6)
        text_features = sum_hidden / lengths
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.float()

    # ---------------- Fusion ----------------
    def fuse_features(self, img_features, q_features):
        x = torch.cat([img_features, q_features], dim=-1)
        gate = torch.sigmoid(self.gate_layer(x))    # (B,H)
        fused = gate * img_features + (1-gate) * q_features

        # optional refinement
        fused = self.fusion(torch.cat([fused, x], dim=-1))
        return fused

    # ---------------- Forward ----------------
    def forward(self, images, questions, answer_input_ids=None):
        img_features = self.encode_image(images)
        img_features = self.img_proj(img_features)
        img_features = img_features.float()

        q_features = self.encode_question(questions["input_ids"], questions["attention_mask"])
        q_features = self.q_proj(q_features)
        q_features = q_features.float()

        batch_size = img_features.size(0)
        context = self.fuse_features(img_features, q_features)

        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device, dtype=torch.float)

        if answer_input_ids is not None:
            # Teacher forcing
            seq_len = answer_input_ids.size(1)
            embeddings = self.embedding(answer_input_ids)
            context_expanded = context.unsqueeze(1).expand(-1, seq_len, -1)
            gru_input = torch.cat([embeddings, context_expanded], dim=-1).float()
            gru_output, hidden = self.gru(gru_input, hidden)
            gru_output = self.ln_gru(gru_output)
            return self.output(gru_output)
        else:
            # Auto-regressive inference
            generated = torch.full((batch_size, self.answer_max_len), self.pad_token_id,
                                   dtype=torch.long, device=self.device)
            generated[:, 0] = self.bos_token_id
            for t in range(1, self.answer_max_len):
                current_input = generated[:, t-1]
                embeddings = self.embedding(current_input).unsqueeze(1).float()
                context_expanded = context.unsqueeze(1)
                gru_input = torch.cat([embeddings, context_expanded], dim=-1).float()
                gru_output, hidden = self.gru(gru_input, hidden)
                gru_output = self.ln_gru(gru_output)
                logits = self.output(gru_output.squeeze(1))
                next_tokens = logits.argmax(dim=-1)
                generated[:, t] = next_tokens
                if (next_tokens == self.eos_token_id).all():
                    break
            return generated
