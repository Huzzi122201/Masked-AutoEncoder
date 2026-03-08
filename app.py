import os
import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import math
import io
from PIL import Image
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="MAE — Masked Autoencoder",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  GLOBAL CSS — dark editorial aesthetic
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

/* ── reset & base ── */
html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}
.stApp {
    background: #0d0d0f;
    color: #e8e4dc;
}

/* ── sidebar ── */
[data-testid="stSidebar"] {
    background: #111115 !important;
    border-right: 1px solid #1e1e24 !important;
}
[data-testid="stSidebar"] * { color: #c8c4bc !important; }
[data-testid="stSidebar"] .stSlider label { color: #7a7a8a !important; font-size: 12px !important; letter-spacing: 1.5px !important; text-transform: uppercase !important; }

/* ── headers ── */
h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; }

/* ── file uploader ── */
[data-testid="stFileUploaderDropzone"] {
    background: #111115 !important;
    border: 2px dashed #2a2a35 !important;
    border-radius: 12px !important;
    transition: border-color 0.3s;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: #e8455a !important;
}

/* ── buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #e8455a, #c0392b) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 13px !important;
    letter-spacing: 1px !important;
    padding: 12px 32px !important;
    width: 100% !important;
    transition: all 0.3s ease !important;
    text-transform: uppercase !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(232, 69, 90, 0.35) !important;
}
.stButton > button:active { transform: translateY(0px) !important; }

/* ── sliders ── */
.stSlider [data-baseweb="slider"] { padding: 4px 0; }
.stSlider [data-baseweb="thumb"] { background: #e8455a !important; border-color: #e8455a !important; }
.stSlider [data-baseweb="track-fill"] { background: #e8455a !important; }

/* ── metric cards ── */
[data-testid="stMetric"] {
    background: #111115;
    border: 1px solid #1e1e24;
    border-radius: 10px;
    padding: 16px 20px;
}
[data-testid="stMetricLabel"] { color: #7a7a8a !important; font-size: 11px !important; letter-spacing: 1.5px !important; text-transform: uppercase !important; }
[data-testid="stMetricValue"] { color: #e8455a !important; font-family: 'Space Mono', monospace !important; font-size: 28px !important; }

/* ── dividers ── */
hr { border-color: #1e1e24 !important; }

/* ── spinner ── */
.stSpinner > div { border-top-color: #e8455a !important; }

/* ── captions / info ── */
.stCaption { color: #4a4a5a !important; font-family: 'Space Mono', monospace !important; font-size: 11px !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  MODEL DEFINITION  (copied from notebook)
# ─────────────────────────────────────────────

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, num_patches: int, embed_dim: int):
        super().__init__()
        pe       = torch.zeros(num_patches, embed_dim)
        position = torch.arange(0, num_patches, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class PatchEmbedding(nn.Module):
    def __init__(self, image_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.patch_size  = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.proj        = nn.Linear(patch_size * patch_size * in_channels, embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        p  = self.patch_size
        h, w = H // p, W // p
        x  = x.view(B, C, h, p, w, p)
        x  = x.permute(0, 2, 4, 3, 5, 1)
        x  = x.reshape(B, h * w, p * p * C)
        return self.proj(x)


def random_masking(x, mask_ratio=0.75):
    B, N, D    = x.shape
    len_keep   = int(N * (1 - mask_ratio))
    noise      = torch.rand(B, N, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    ids_keep    = ids_shuffle[:, :len_keep].to(x.device)
    ids_restore = ids_restore.to(x.device)
    x_masked    = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
    mask        = torch.ones([B, N], device=x.device)
    mask[:, :len_keep] = 0
    mask        = torch.gather(mask, dim=1, index=ids_restore)
    return x_masked, mask, ids_restore


class MAEEncoder(nn.Module):
    def __init__(self, img_size=224, patch_size=16, embed_dim=768,
                 num_layers=12, num_heads=12):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, embed_dim=embed_dim)
        n_patches        = (img_size // patch_size) ** 2
        self.pos_enc     = SinusoidalPositionalEncoding(n_patches, embed_dim)
        enc_layer        = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 4, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

    def forward(self, x, mask_ratio=0.75):
        x = self.patch_embed(x)
        x = self.pos_enc(x)
        x_masked, mask, ids_restore = random_masking(x, mask_ratio)
        x_enc = self.encoder(x_masked)
        return x_enc, mask, ids_restore, x_masked


class MAEDecoder(nn.Module):
    def __init__(self, n_patches, patch_size=16, enc_dim=768,
                 dec_dim=384, num_layers=12, num_heads=6):
        super().__init__()
        self.n_patches     = n_patches
        self.patch_size    = patch_size
        self.mask_token    = nn.Parameter(torch.zeros(1, 1, dec_dim))
        self.decoder_embed = nn.Linear(enc_dim, dec_dim)
        self.pos_enc       = SinusoidalPositionalEncoding(n_patches, dec_dim)
        dec_layer          = nn.TransformerEncoderLayer(
            d_model=dec_dim, nhead=num_heads,
            dim_feedforward=dec_dim * 4, batch_first=True
        )
        self.decoder      = nn.TransformerEncoder(dec_layer, num_layers=num_layers)
        self.decoder_pred = nn.Linear(dec_dim, patch_size * patch_size * 3)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(self, x_enc, ids_restore):
        B  = x_enc.size(0)
        x  = self.decoder_embed(x_enc)
        mt = self.mask_token.expand(B, self.n_patches - x.size(1), -1)
        x_ = torch.cat([x, mt], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).expand_as(x_))
        x_ = self.pos_enc(x_)
        x_ = self.decoder(x_)
        return self.decoder_pred(x_)


class MAE(nn.Module):
    def __init__(self, img_size=224, patch_size=16,
                 enc_dim=768, dec_dim=384,
                 enc_layers=12, dec_layers=12,
                 enc_heads=12, dec_heads=6, mask_ratio=0.75):
        super().__init__()
        self.n_patches  = (img_size // patch_size) ** 2
        self.mask_ratio = mask_ratio
        self.encoder    = MAEEncoder(img_size, patch_size, enc_dim, enc_layers, enc_heads)
        self.decoder    = MAEDecoder(self.n_patches, patch_size, enc_dim, dec_dim, dec_layers, dec_heads)

    def forward(self, x):
        x_enc, mask, ids_restore, _ = self.encoder(x, mask_ratio=self.mask_ratio)
        x_rec = self.decoder(x_enc, ids_restore)
        return x_rec, mask


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────

def patchify(imgs, patch_size=16):
    B, C, H, W = imgs.shape
    h = w = H // patch_size
    x = imgs.reshape(B, C, h, patch_size, w, patch_size)
    x = x.permute(0, 2, 4, 3, 5, 1)
    return x.reshape(B, h * w, patch_size * patch_size * C)


def unpatchify(patches, patch_size=16, img_size=224):
    B, N, D = patches.shape
    h = w = img_size // patch_size
    x = patches.reshape(B, h, w, patch_size, patch_size, 3)
    x = x.permute(0, 5, 1, 3, 2, 4)
    return x.reshape(B, 3, h * patch_size, w * patch_size)


def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (tensor.cpu() * std + mean).clamp(0, 1)


def build_masked_image(imgs, mask, patch_size=16):
    patches        = patchify(imgs, patch_size)
    masked_patches = patches.clone()
    for b in range(patches.shape[0]):
        masked_patches[b][mask[b] == 1] = 0.5
    return unpatchify(masked_patches, patch_size, imgs.shape[-1])


def tensor_to_pil(tensor):
    arr = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


@st.cache_resource
def load_model(weights_path: str, device: str):
    model = MAE()
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model.to(device)


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style='padding: 8px 0 24px 0;'>
        <div style='font-family:"Space Mono",monospace; font-size:10px;
                    letter-spacing:3px; text-transform:uppercase;
                    color:#4a4a5a; margin-bottom:8px;'>
            FAST NUCES · AI4009
        </div>
        <div style='font-size:22px; font-weight:700; color:#e8e4dc;
                    line-height:1.2;'>
            Masked<br>Autoencoder
        </div>
        <div style='width:32px; height:3px; background:#e8455a; margin-top:10px;'></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("##### ⚙️ Configuration")

    mask_ratio = st.slider(
        "MASKING RATIO",
        min_value=0.1, max_value=0.95,
        value=0.75, step=0.05,
        help="Fraction of image patches to hide from the model"
    )

    st.markdown(f"""
    <div style='background:#0d0d0f; border:1px solid #1e1e24;
                border-radius:8px; padding:14px 16px; margin:12px 0;
                font-family:"Space Mono",monospace; font-size:12px;'>
        <div style='color:#4a4a5a; font-size:10px; letter-spacing:1px;
                    text-transform:uppercase; margin-bottom:8px;'>Patch Stats</div>
        <div style='color:#e8e4dc;'>Visible &nbsp; <span style='color:#4ade80;'>
            {int(196*(1-mask_ratio))} patches</span></div>
        <div style='color:#e8e4dc;'>Masked &nbsp;&nbsp; <span style='color:#e8455a;'>
            {int(196*mask_ratio)} patches</span></div>
        <div style='color:#e8e4dc;'>Total &nbsp;&nbsp;&nbsp;&nbsp; 
            <span style='color:#7a7a8a;'>196 patches</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-family:"Space Mono",monospace; font-size:10px;
                color:#2a2a35; line-height:1.8;'>
        ViT-Base Encoder · 768d · 12L<br>
        ViT-Small Decoder · 384d · 12L<br>
        Patch Size 16×16 · Image 224×224<br>
        ~108M Parameters
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  MAIN HEADER
# ─────────────────────────────────────────────

st.markdown("""
<div style='padding: 32px 0 8px 0;'>
    <div style='font-family:"Space Mono",monospace; font-size:11px;
                letter-spacing:3px; text-transform:uppercase;
                color:#4a4a5a; margin-bottom:12px;'>
        Self-Supervised Visual Learning
    </div>
    <h1 style='font-size:clamp(28px,4vw,48px); font-weight:700;
               color:#e8e4dc; line-height:1.1; margin:0;'>
        See What's <span style='color:#e8455a;'>Hidden</span>
    </h1>
    <p style='color:#4a4a5a; font-size:15px; margin-top:10px; max-width:560px;'>
        Upload any image. The MAE reconstructs it from only
        <strong style='color:#e8e4dc;'>{:.0f}%</strong> of visible patches —
        filling in the missing <strong style='color:#e8455a;'>{:.0f}%</strong> from learned visual knowledge.
    </p>
</div>
<hr>
""".format((1 - mask_ratio) * 100, mask_ratio * 100), unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  UPLOAD ZONE
# ─────────────────────────────────────────────

col_upload, col_spacer = st.columns([1.6, 1])
with col_upload:
    uploaded = st.file_uploader(
        "Drop your image here",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed"
    )
    st.caption("Accepts JPG · PNG · WEBP · any size (auto-resized to 224×224)")


# ─────────────────────────────────────────────
#  INFERENCE
# ─────────────────────────────────────────────

if uploaded is not None:

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── load model from checkpoint placed next to app.py ──
    CHECKPOINT = "best_mae_model.pth"
    model = None
    if os.path.exists(CHECKPOINT):
        try:
            model = load_model(CHECKPOINT, device)
        except Exception as e:
            st.error(f"Could not load checkpoint `{CHECKPOINT}`: {e}")
    else:
        model = MAE(mask_ratio=mask_ratio).to(device)
        model.eval()
        st.warning(f"⚠️  `{CHECKPOINT}` not found next to `app.py` — running with **random weights** (demo mode).")

    if model is not None:
        # ── preprocess ──
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        pil_img = Image.open(uploaded).convert("RGB")
        img_tensor = transform(pil_img).unsqueeze(0).to(device)

        # ── run model with chosen mask ratio ──
        # Temporarily patch mask_ratio
        model.mask_ratio = mask_ratio
        model.encoder.pos_enc  # warmup

        with st.spinner("Running MAE inference…"):
            with torch.no_grad():
                recon_patches, mask = model(img_tensor)

        recon_imgs  = unpatchify(recon_patches, 16, 224)
        masked_imgs = build_masked_image(img_tensor.cpu(), mask.cpu(), 16)

        orig_np   = denormalize(img_tensor[0].cpu()).permute(1, 2, 0).numpy()
        masked_np = denormalize(masked_imgs[0].cpu()).permute(1, 2, 0).numpy()
        recon_np  = denormalize(recon_imgs[0].cpu()).permute(1, 2, 0).numpy()

        # ── metrics ──
        psnr_val = psnr_metric(orig_np, recon_np, data_range=1.0)
        ssim_val = ssim_metric(orig_np, recon_np, data_range=1.0, channel_axis=2)
        mse_val  = float(np.mean((orig_np - recon_np) ** 2))

        # ── RESULTS LAYOUT ──

        st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

        # Metrics row
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("PSNR", f"{psnr_val:.2f} dB",
                  help="Peak Signal-to-Noise Ratio — higher is better")
        m2.metric("SSIM", f"{ssim_val:.4f}",
                  help="Structural Similarity — 1.0 is perfect")
        m3.metric("MSE", f"{mse_val:.5f}",
                  help="Mean Squared Error — lower is better")
        m4.metric("Mask Ratio", f"{mask_ratio*100:.0f}%",
                  help="Fraction of patches hidden from encoder")

        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)

        # Three-panel image display
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("""
            <div style='font-family:"Space Mono",monospace; font-size:10px;
                        letter-spacing:2px; text-transform:uppercase;
                        color:#4ade80; margin-bottom:10px;'>
                ① Original Input
            </div>""", unsafe_allow_html=True)
            st.image(tensor_to_pil(torch.tensor(orig_np).permute(2, 0, 1)),
                     use_column_width=True)

        with c2:
            st.markdown(f"""
            <div style='font-family:"Space Mono",monospace; font-size:10px;
                        letter-spacing:2px; text-transform:uppercase;
                        color:#e8455a; margin-bottom:10px;'>
                ② Masked ({mask_ratio*100:.0f}% hidden)
            </div>""", unsafe_allow_html=True)
            st.image(tensor_to_pil(torch.tensor(masked_np).permute(2, 0, 1)),
                     use_column_width=True)

        with c3:
            st.markdown("""
            <div style='font-family:"Space Mono",monospace; font-size:10px;
                        letter-spacing:2px; text-transform:uppercase;
                        color:#7c3aed; margin-bottom:10px;'>
                ③ Reconstruction
            </div>""", unsafe_allow_html=True)
            st.image(tensor_to_pil(torch.tensor(recon_np).permute(2, 0, 1)),
                     use_column_width=True)

        # Patch visualisation bar
        st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
        st.markdown("""
        <div style='font-family:"Space Mono",monospace; font-size:10px;
                    letter-spacing:2px; text-transform:uppercase;
                    color:#4a4a5a; margin-bottom:10px;'>Patch Map</div>
        """, unsafe_allow_html=True)

        mask_np   = mask[0].cpu().numpy()
        patch_row = ""
        for i, m in enumerate(mask_np):
            col  = "#e8455a" if m == 1 else "#4ade80"
            patch_row += f"<span style='display:inline-block;width:10px;height:10px;background:{col};margin:1px;border-radius:1px;'></span>"
        st.markdown(f"<div style='line-height:0;'>{patch_row}</div>", unsafe_allow_html=True)
        st.markdown("""
        <div style='margin-top:8px; font-family:"Space Mono",monospace; font-size:10px; color:#2a2a35;'>
            <span style='color:#4ade80;'>■</span> visible &nbsp;&nbsp;
            <span style='color:#e8455a;'>■</span> masked
        </div>
        """, unsafe_allow_html=True)

        # Download button
        st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
        recon_pil = tensor_to_pil(torch.tensor(recon_np).permute(2, 0, 1))
        buf = io.BytesIO()
        recon_pil.save(buf, format="PNG")
        st.download_button(
            label="⬇  Download Reconstruction",
            data=buf.getvalue(),
            file_name="mae_reconstruction.png",
            mime="image/png",
        )

else:
    # ── Empty state ──
    st.markdown("""
    <div style='text-align:center; padding:80px 40px;
                border:2px dashed #1e1e24; border-radius:16px;
                margin-top:32px;'>
        <div style='font-size:48px; margin-bottom:16px;'>🎭</div>
        <div style='font-family:"Space Mono",monospace; font-size:13px;
                    color:#4a4a5a; letter-spacing:1px;'>
            Upload an image above to begin reconstruction
        </div>
        <div style='margin-top:24px; display:flex; justify-content:center; gap:32px;
                    font-family:"Space Mono",monospace; font-size:11px; color:#2a2a35;'>
            <span>① Upload Image</span>
            <span>→</span>
            <span>② Set Mask Ratio</span>
            <span>→</span>
            <span>③ See Reconstruction</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown("<div style='height:48px'></div>", unsafe_allow_html=True)
st.markdown("""
<hr>
<div style='font-family:"Space Mono",monospace; font-size:10px;
            color:#2a2a35; text-align:center; padding:16px 0;
            letter-spacing:1px;'>
    FAST NUCES · Generative AI AI4009 · Spring 2026 &nbsp;·&nbsp;
    MAE: He et al. CVPR 2022 &nbsp;·&nbsp;
    ViT-Base Encoder / ViT-Small Decoder
</div>
""", unsafe_allow_html=True)
